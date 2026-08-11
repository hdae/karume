"""op 契約テーブル — op ごとの「アリティ / dtype 規則 / attrs スキーマ」を Python 側に
1 箇所で持つ。

TS 側の正本は packages/runtime/src/ops.ts で、本表はその同義物。エクスポータの emit 集合
⊆ ランタイム実行可能集合を dtype 込みで突合するのが目的（op 名だけの突合は dtype 差を見逃す —
ADR 0005）。両者が割れると「export は緑、ブラウザだけ落ちる」になるため、
op を足すときは TS 契約表・本表・golden を 1 セットで動かす。

NOTE: 出力 shape の導出（broadcast/縮約）は本表に持たない — 束縛解決後に shape を
計算する消費者が Python 側に無いため（ランタイムは TS の packages/runtime/src/ops.ts が持つ）。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from karume.dims import MAX_SAFE_INT, is_symbol_name
from karume.ir import IrNode

#: 単項 elementwise。attrs を持つ 6 本（clamp / clamp_min / leaky_relu / 比較 3 本）も、
#: 出力 dtype が入力と違う比較 3 本もここに属する — kind が意味するのは「入力 1 本の
#: elementwise」だけで、attrs スキーマと出力 dtype は契約表の別の欄が持つ
#: （packages/runtime/src/ops.ts と同義）。
UNARY_OPS = (
    "neg",
    "abs",
    "exp",
    "log",
    "log1p",
    "sqrt",
    "tanh",
    "sigmoid",
    "relu",
    "gelu",
    # torch の approximate="tanh" 形。erf 形の `gelu` とは値が違うので別 op で表す
    # （attrs 空の契約に近似種別を載せる欄が無い）。
    "gelu_tanh",
    "bitwise_not",
    "clamp",
    # 片側 clamp（ADR 0017）。既存 clamp の attrs を optional 化して兼ねる案は
    # 「宣言済み attrs の既定値補完はしない」（ADR 0012）を崩すので採らない。
    "clamp_min",
    "leaky_relu",
    "ge_scalar",
    "le_scalar",
    "gt_scalar",
)
#: 二項 elementwise（torch 準拠の右詰め broadcast）。`ge` は f32 × f32 → **bool**。
BINARY_OPS = ("add", "sub", "mul", "div", "ge", "bitwise_and")
#: 1 軸の reduce（attrs `dim`・keepdim 無し）。最終次元は行カーネル、それ以外は軸変種で
#: 実行する（docs/research/2026-08-04-vae-axis-reduce-recon.md §5）。
REDUCE_OPS = ("sum", "amax", "amin")
#: 三項 elementwise `out = cond ? a : b`（torch の where）。3 者とも右詰め broadcast で、
#: 条件が先頭スロットでも**出力は値の側**（写像 bool → f32 — _OUTPUT_DTYPES）。
WHERE_OP = "where"
#: 最終次元の前縁和（attrs `dim`、最終次元のみ — softmax と同じ絞り方）。
CUMSUM_OP = "cumsum"
MATMUL_OP = "matmul"
#: バッチ matmul（ADR 0012）。`[B,M,K] × [B,K,N] → [B,M,N]` の **rank-3 のみ**で、
#: rank-2 は matmul の担当（兼用にしない — 契約の shape 規則が rank 多相になる）。
BMM_OP = "bmm"
#: 最終次元の gather（ADR 0012）。`out[..., j] = src[..., index[..., j]]`。軸は最終次元固定で
#: attrs を持たない（実測は dim = -1 のみ）。入力スロットで dtype が違う唯一の op。
GATHER_OP = "gather"
#: 意味論 dtype 変換（attrs `to` が変換先 — ADR 0009）。
CAST_OP = "cast"

#: レイアウト op（ADR 0011）。reshape は要素順を変えない（view / squeeze / unsqueeze の
#: 正規化先）で実行はバッファ別名、permute / expand は strided 実体化コピー 1 カーネル族。
RESHAPE_OP = "reshape"
PERMUTE_OP = "permute"
EXPAND_OP = "expand"

#: レイアウト op 第 2 群（ADR 0014）。full-write 不変条件（全カーネルが出力の全バイトを書く）を
#: 満たす形だけを語彙に入れる。
#:
#: - `slice` — 静的軸・静的範囲の切り出し（記号軸は sym_prefix_slice の担当 — 重複させない）。
#:   実行は strided **読み**コピー族の流用で、可変点は params の offset 1 語。
#: - `cat` — 静的軸の連結。**入力数が可変**（OpContract.variadic）で、実行は strided **書き**
#:   コピー族。全入力で出力全域を覆うのでノード単位では full-write が成立する。
#: - `pad` — 最終次元の定数 0 埋め。専用カーネル 1 本が出力全域を書く。
#: - `flip` — 静的軸の添字反転。専用の極小カーネル。
SLICE_OP = "slice"
CAT_OP = "cat"
PAD_OP = "pad"
FLIP_OP = "flip"

#: 記号 prefix スライス（ADR 0010）。Tmax で焼いた定数から束縛後の `coeff·sym+offset` 長の
#: 先頭を切り出す。入力は記号を含まない静的 shape（= Tmax 形）で、実行は strided
#: 実体化コピーの流用（新カーネル 0）。
SYM_PREFIX_SLICE_OP = "sym_prefix_slice"

#: 融合 op（ADR 0012 / ADR 0015 / ADR 0017）。ADR 0007 の「分解禁止 10 op」は全て
#: カーネルを持つ（conv2d は Anima の VAE decoder で実測に出た — ADR 0017）。
#: bias / affine を持つ 4 本は**アリティ固定**で、bias 無しの conv はエクスポータの
#: ゼロ bias 合成でアリティ 3 へ正規化される（arity 分岐を語彙に持ち込まない — ADR 0015）。
LINEAR_OP = "linear"
LAYER_NORM_OP = "layer_norm"
#: RMSNorm（ADR 0017）。**アリティ 2**（x, weight — bias が無い）で attrs は eps のみ。
#: 正規化長の正本は weight の長さ（normalized_shape の欄は作らない — 二重管理にしない）。
RMS_NORM_OP = "rms_norm"
SOFTMAX_OP = "softmax"
#: 融合 attention（ADR 0023）。`out = softmax_lastdim((q·scale) @ (k·scale)ᵀ) @ v`。
#: **アリティ 3 固定**（q / k / v）・入力は rank-4 head-first（`[B,H,M,D]` / `[B,H,N,D]` ×2）で
#: **D は 3 者とも同じ**・出力は `[B,H,M,D]`。mask / causal / dropout / GQA は語彙に無く、
#: 該当する SDPA は `convert._h_attention` が全件列挙して fail loudly にする。
#:
#: MUST: `scale` は **q と k の両方に掛かる（半スケール契約）** — torch の
#: `_scaled_dot_product_attention_math` の `√scale_factor` と同義。エクスポータは SDPA の
#: `scale` 引数（省略時 `1/√D`）から `f32(math.sqrt(scale_factor))` を計算して載せる。
ATTENTION_OP = "attention"
EMBEDDING_OP = "embedding"
MASKED_FILL_OP = "masked_fill"
CONV1D_OP = "conv1d"
#: 2 次元畳み込み（ADR 0017）。重みは **[Cout, Cin/groups, Kh, Kw]** で、attrs の
#: stride / padding / dilation は **H/W の 2 成分**（groups はスカラ）。4 つとも宣言必須。
CONV2D_OP = "conv2d"
#: 転置畳み込み（ADR 0015）。重みは **[Cin, Cout, K]**（conv1d の [Cout, Cin/groups, K] と転置）。
#: attrs は stride / padding のみで、受理するのは出力長が L*stride になる形だけ。
CONV_TRANSPOSE1D_OP = "conv_transpose1d"

#: 低精度格納が**適格**になる重みスロット（op 名 → 入力スロット番号 — ADR 0018）。
#: TS 側 `packages/runtime/src/ops.ts` の `WEIGHT_SLOTS` の鏡像で、ずれは適合表
#: （packages/runtime/tests/fixtures/op-contracts.json の `weight_slot`）が両側から
#: 突き合わせて落とす。
#:
#: MUST: bias を含めない。プロトタイプは bias の f32 定数が weight を道連れに降格させて
#: f16 の適格を 0MB にした（ADR 0006 が名指しした根治対象）— 「bias は常に f32」は
#: 「bias スロットを適格判定に載せない」ことでしか担保できない。
WEIGHT_SLOTS: Mapping[str, int] = MappingProxyType(
    {
        LINEAR_OP: 1,
        CONV1D_OP: 1,
        CONV2D_OP: 1,
        CONV_TRANSPOSE1D_OP: 1,
        EMBEDDING_OP: 0,
    }
)

#: per-channel scale の**チャネル軸**（op 名 → 重みテンソルの軸番号 — ADR 0019）。
#: TS 側 `packages/runtime/src/ops.ts` の `WEIGHT_CHANNEL_AXES` の鏡像で、ずれは適合表
#: （packages/runtime/tests/fixtures/op-contracts.json の `channel_axis`）が両側から
#: 突き合わせて落とす。
#:
#: 出力チャネルの軸。linear `[out,in]` / conv1d `[Cout,Cin/g,K]` / conv2d `[Cout,Cin/g,Kh,Kw]` /
#: embedding `[V,H]` は 0 で、**conv_transpose1d だけ `[Cin,Cout,K]` の転置レイアウトで 1**。
#:
#: MUST: キー集合は WEIGHT_SLOTS と一致させる（新しい重みスロットが軸 0 として黙って
#: 量子化されるのを防ぐ）。`quantize.QUANT_CHANNEL_AXES`（モジュール型で引く同じ表）とも
#: 軸が一致していることを test_quantize.py が固定する。
WEIGHT_CHANNEL_AXES: Mapping[str, int] = MappingProxyType(
    {
        LINEAR_OP: 0,
        CONV1D_OP: 0,
        CONV2D_OP: 0,
        CONV_TRANSPOSE1D_OP: 1,
        EMBEDDING_OP: 0,
    }
)

#: strided コピーカーネルが扱える最大 rank（ADR 0011）。エクスポータ側でも見るのは、
#: 超過を export 時点で落とすため（「export は緑、ブラウザだけ落ちる」を作らない）。
STRIDED_RANK = 4

#: IR v1 の意味論 dtype 語彙（docs/ir-v1.md）。
SEMANTIC_DTYPES = frozenset({"f32", "i32", "bool"})
#: f32 専業（実測グラフに i32 / bool 形が現れていない — 対称性のためには解禁しない）。
F32_DTYPES = frozenset({"f32"})
#: mask 外積 mul(i64,i64) と `1 - attention_mask` の sub（recon §3-8）。入力値依存で
#: 定数畳み込みできないため実行系に要る。
F32_I32_DTYPES = frozenset({"f32", "i32"})
BOOL_DTYPES = frozenset({"bool"})
#: bool の行 sum（真の個数 — 出力は i32）。sdp の searchsorted（recon §2）。
F32_BOOL_DTYPES = frozenset({"f32", "bool"})
#: 整数添字専業（gather の index スロット — 実測は i32[16,T,T]）。
I32_DTYPES = frozenset({"i32"})
#: グラフ入力として転送できる意味論 dtype（ADR 0009 — 要素は全型 4 バイト）。
IO_DTYPES = SEMANTIC_DTYPES
#: ランタイムが実行できる格納 dtype（宣言としては bf16 も valid）。
#: `i32` は記号依存定数の焼き込み先として実行対象（生の int32 — ADR 0010）。
#: `f16` は ADR 0018 / `i8` は ADR 0019 — 適格な重みスロットは圧縮のまま GPU 常駐し、
#: 適格外はロード時に CPU で f32 展開されるので、どちらの経路でも実行できる
#: （TS 側 RuntimeSupport.storage の鏡像）。
M0_STORAGE_DTYPES = frozenset({"f32", "f16", "i8", "i32"})

OpKind = Literal[
    "unary",
    "binary",
    "where",
    "cumsum",
    "matmul",
    "bmm",
    "gather",
    "row_reduce",
    "cast",
    "reshape",
    "permute",
    "expand",
    "slice",
    "cat",
    "pad",
    "flip",
    "sym_prefix_slice",
    "linear",
    "layer_norm",
    "rms_norm",
    "softmax",
    "attention",
    "embedding",
    "masked_fill",
    "conv1d",
    "conv2d",
    "conv_transpose1d",
]

#: attr キー → 値の検査関数（ADR 0012）。宣言したキーは全て必須で、宣言外は fail loudly。
AttrSchema = Mapping[str, Callable[[Any, str], None]]


class OpContractError(Exception):
    """op 契約に反する IR（契約表に無い op・アリティ違反・契約外 attrs・dtype 違反）。"""


def _assert_cast_target(value: Any, where: str) -> str:
    """cast の変換先。

    MUST: f32 → i32 は torch 準拠の truncate（0 方向切り捨て）、x → bool は x != 0。
    bool の実表現は u32 の 0 / 1（ADR 0009）。丸め規約は契約に書く（黙って近似しない）。
    """
    if not isinstance(value, str) or value not in SEMANTIC_DTYPES:
        raise OpContractError(
            f"{where}: cast の変換先が意味論 dtype でない"
            f"（{' / '.join(sorted(SEMANTIC_DTYPES))}）: {value!r}"
        )
    return value


CAST_ATTRS: AttrSchema = {"to": lambda value, where: _assert_cast_target(value, where)}


def _assert_permute_dims(value: Any, where: str) -> list[int]:
    """permute の軸並べ替え表（`dims[d]` = 出力の次元 d が取る入力の次元番号）。

    MUST: 負の軸番号を受理しない（torch の -1 表記はエクスポータ境界で正規化する規約）。
    同じ並べ替えに 2 通りの IR ができると CSE も突合も割れる。
    MUST: 重複を拒否する（並べ替えは全単射）。入力 rank との突合はランタイムの shape 計算側。
    """
    if not isinstance(value, list) or not value:
        raise OpContractError(f"{where}: permute の dims が非空のリストでない: {value!r}")
    for dim in value:
        # 上限は安全整数（TS 側の `Number.isSafeInteger` と同じ受理集合 — 任意精度の int を
        # 通すと「エクスポータは書けるがランタイムが受理しない dims」ができる）。
        if isinstance(dim, bool) or not isinstance(dim, int) or not 0 <= dim <= MAX_SAFE_INT:
            raise OpContractError(f"{where}: permute の dims に非負整数でない要素がある: {value!r}")
    if len(set(value)) != len(value):
        raise OpContractError(f"{where}: permute の dims {value!r} に重複がある")
    return list(value)


PERMUTE_ATTRS: AttrSchema = {"dims": lambda value, where: _assert_permute_dims(value, where)}


def permute_dims(attrs: Mapping[str, Any], where: str) -> list[int]:
    """permute ノードの軸並べ替え表（検査は assert_node_contract が済ませている前提）。"""
    return _assert_permute_dims(attrs.get("dims"), f"{where} の attrs.dims")


#: slice の切り出し指定（ADR 0014）。`dim` 軸を `[start, end)` に縮める。
#:
#: MUST: 3 つとも非負整数（負の軸表記・負の添字表記・torch 既定の巨大 end は境界で正規化する
#: 規約 — permute の dims と同じ）。`start <= end` と `end <= 軸長` は shapes 層が持つ
#: （キーと入力 shape を跨ぐ規則は attrs スキーマでは表せない — clamp の min/max と同じ分担）。
SLICE_ATTRS: AttrSchema = {
    "dim": lambda value, where: _assert_integer_attr(value, where, 0),
    "start": lambda value, where: _assert_integer_attr(value, where, 0),
    "end": lambda value, where: _assert_integer_attr(value, where, 0),
}

#: cat / flip の軸（非負の軸番号。入力 rank との突合は shapes 層）。
AXIS_ATTRS: AttrSchema = {"dim": lambda value, where: _assert_integer_attr(value, where, 0)}

#: pad の左右パディング幅（**最終次元・定数 0 のみ**）。
#:
#: MUST: 負幅（torch の constant_pad_nd は負で切り詰めできる）を受理しない — 切り詰めは
#: slice の意味で、通すと同じ形を 2 つの op で書けるうえ出力長の計算が負になる。
#: MUST: 埋め値の欄を作らない（実測は 0 のみ — conv1d の groups と同じ絞り方）。
PAD_ATTRS: AttrSchema = {
    "left": lambda value, where: _assert_integer_attr(value, where, 0),
    "right": lambda value, where: _assert_integer_attr(value, where, 0),
}


def slice_attrs(attrs: Mapping[str, Any], where: str) -> tuple[int, int, int]:
    """slice ノードの (dim, start, end)（検査は assert_node_contract が済ませている前提）。"""
    return (
        _assert_integer_attr(attrs.get("dim"), f"{where} の attrs.dim", 0),
        _assert_integer_attr(attrs.get("start"), f"{where} の attrs.start", 0),
        _assert_integer_attr(attrs.get("end"), f"{where} の attrs.end", 0),
    )


def axis_dim(attrs: Mapping[str, Any], where: str) -> int:
    """cat / flip ノードの軸（非負の軸番号）。"""
    return _assert_integer_attr(attrs.get("dim"), f"{where} の attrs.dim", 0)


def pad_attrs(attrs: Mapping[str, Any], where: str) -> tuple[int, int]:
    """pad ノードの (left, right)。"""
    return (
        _assert_integer_attr(attrs.get("left"), f"{where} の attrs.left", 0),
        _assert_integer_attr(attrs.get("right"), f"{where} の attrs.right", 0),
    )


#: sym_prefix_slice の 1 軸ぶんの切り出し指定のキー（`dim` を長さ `coeff·sym+offset` に縮める）。
PREFIX_SLICE_KEYS = ("dim", "coeff", "offset")


def _assert_prefix_sym(value: Any, where: str) -> str:
    """sym_prefix_slice の `sym`（次元言語のシンボル名）。

    束縛済みかどうかはグラフを見ないと分からないので、ここは綴りだけを見る
    （graph.symbols との突合は verify.assert_op_contracts）。
    """
    if not isinstance(value, str) or not is_symbol_name(value):
        raise OpContractError(f"{where}: sym_prefix_slice の sym がシンボル名でない: {value!r}")
    return value


def _assert_prefix_slices(value: Any, where: str) -> list[dict[str, int]]:
    """sym_prefix_slice の `slices`。

    MUST: 軸の重複を拒否する（同じ軸に 2 つの指定があると片方が黙って消え、宣言 shape との
    照合だけが通る形が作れる）。係数 1 以上・オフセット 0 以上は次元言語と同じ値域。
    """
    if not isinstance(value, list) or not value:
        raise OpContractError(
            f"{where}: sym_prefix_slice の slices が非空のリストでない: {value!r}"
        )
    slices: list[dict[str, int]] = []
    for index, raw in enumerate(value):
        at = f"{where}[{index}]"
        if not isinstance(raw, dict):
            raise OpContractError(f"{at}: オブジェクトでない: {raw!r}")
        unknown = sorted(key for key in raw if key not in PREFIX_SLICE_KEYS)
        if unknown:
            raise OpContractError(f"{at}: 未知のキー {unknown}")
        for key in PREFIX_SLICE_KEYS:
            if key not in raw:
                raise OpContractError(f"{at}: キー '{key}' が無い")
        slices.append(
            {
                "dim": _assert_integer_attr(raw["dim"], f"{at}.dim", 0),
                "coeff": _assert_integer_attr(raw["coeff"], f"{at}.coeff", 1),
                "offset": _assert_integer_attr(raw["offset"], f"{at}.offset", 0),
            }
        )
    dims = [entry["dim"] for entry in slices]
    if len(set(dims)) != len(dims):
        raise OpContractError(f"{where}: sym_prefix_slice の slices に同じ dim が 2 度ある")
    return slices


SYM_PREFIX_SLICE_ATTRS: AttrSchema = {
    "sym": lambda value, where: _assert_prefix_sym(value, where),
    "slices": lambda value, where: _assert_prefix_slices(value, where),
}


def sym_prefix_slice_attrs(
    attrs: Mapping[str, Any], where: str
) -> tuple[str, list[dict[str, int]]]:
    """sym_prefix_slice ノードの attrs（検査は assert_node_contract が済ませている前提）。"""
    return (
        _assert_prefix_sym(attrs.get("sym"), f"{where} の attrs.sym"),
        _assert_prefix_slices(attrs.get("slices"), f"{where} の attrs.slices"),
    )


def _assert_integer_attr(value: Any, where: str, minimum: int) -> int:
    """`minimum` 以上の整数 attr。

    MUST: `bool` を除く（Python の bool は int の派生で、`True` が 1 として素通りする）。
    MUST: 安全整数の上限も見る（TS 側は `Number.isSafeInteger` で暗黙にこの上限を持つ）。
    Python の int は任意精度なので、上限を書かないと「エクスポータは書けるがランタイムが
    受理しない attr 値」が作れる — 受理集合は両側で同じでなければならない。
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= MAX_SAFE_INT
    ):
        raise OpContractError(f"{where}: {minimum} 以上の整数でない: {value!r}")
    return value


def _assert_normalized_shape(value: Any, where: str) -> list[int]:
    """layer_norm の正規化軸。**長さ 1（= 最終次元）のみ**受理する。

    MUST: 多軸正規化を「対称性のため」受け入れない（ADR 0007 の語彙 allowlist 凍結）。
    実測は全 7 本が [1024]（recon §5）で、行カーネルは最終次元の連続並びが前提。
    """
    if not isinstance(value, list) or len(value) != 1:
        raise OpContractError(
            f"{where}: layer_norm の normalized_shape は長さ 1 のリストのみ"
            f"（最終次元の正規化だけを実行できる）: {value!r}"
        )
    return [_assert_integer_attr(value[0], f"{where}[0]", 1)]


def _assert_eps(value: Any, where: str, what: str) -> float:
    """正規化 op（layer_norm / rms_norm）の eps。

    **有限の正数**のみ（0 は分散 0・全要素 0 の行で 1/sqrt(0) を作る）。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpContractError(f"{where}: {what} の eps が数値でない: {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise OpContractError(f"{where}: {what} の eps は有限の正数でない: {value!r}")
    return number


def _assert_finite_attr(value: Any, where: str, what: str) -> float:
    """params の f32 語で運ぶスカラ attr。**有限の f32 スカラ**（IR v1 は非有限値を拒否する）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpContractError(f"{where}: {what} が数値でない: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise OpContractError(f"{where}: {what} は有限の数値でない: {value!r}")
    return number


def _assert_fill_value(value: Any, where: str) -> float:
    """masked_fill の埋め値。

    実測値 -3.4028234663852886e+38 は f32 の最小有限値ちょうどで、JSON 往復でも f32 丸めでも
    ulp が動かない（tests/test_convert.py が固定）。
    """
    return _assert_finite_attr(value, where, "masked_fill の value")


LAYER_NORM_ATTRS: AttrSchema = {
    "normalized_shape": lambda value, where: _assert_normalized_shape(value, where),
    "eps": lambda value, where: _assert_eps(value, where, "layer_norm"),
}

#: rms_norm の attrs（ADR 0017）。
#:
#: MUST: normalized_shape の欄を作らない。正規化軸は最終次元 1 本に固定で、長さは
#: **weight の長さ**が持つ（layer_norm は attrs と weight で同じ事実を二重に持っている）。
RMS_NORM_ATTRS: AttrSchema = {"eps": lambda value, where: _assert_eps(value, where, "rms_norm")}

#: 負の軸表記はエクスポータ境界で正規化する規約（permute の dims と同じ）。最終次元との
#: 突合は rank が分かるランタイムの shape 計算側。
SOFTMAX_ATTRS: AttrSchema = {"dim": lambda value, where: _assert_integer_attr(value, where, 0)}

#: attention の attrs（ADR 0023）。
#:
#: MUST: `scale` は宣言必須で既定値補完をしない（ランタイム側と同じ規律）。省略を許すと
#: 「scale を明示した SDPA」と「補完された IR」が同じ形になり、取り違えが値にしか出ない。
#: MUST: mask / causal / dropout の欄を作らない — 欄が無いこと自体が「語彙に無い」を構造で表す。
ATTENTION_ATTRS: AttrSchema = {
    "scale": lambda value, where: _assert_finite_attr(value, where, "attention の scale")
}

#: torch の padding_idx は**受理するが forward には効かない**（勾配で padding 行を更新しない
#: ための欄で、順伝播は素の行 gather と同じ）。無視するために契約から落とすと「未知 attr は
#: fail loudly」の規律に穴が開くので、値域（-1 = 未指定の番兵）だけ検査して運ぶ。
EMBEDDING_ATTRS: AttrSchema = {
    "padding_idx": lambda value, where: _assert_integer_attr(value, where, -1)
}

MASKED_FILL_ATTRS: AttrSchema = {"value": lambda value, where: _assert_fill_value(value, where)}

CLAMP_ATTRS: AttrSchema = {
    "min": lambda value, where: _assert_finite_attr(value, where, "clamp の min"),
    "max": lambda value, where: _assert_finite_attr(value, where, "clamp の max"),
}

#: clamp_min の attrs（ADR 0017 — チャネル L2 正規化の clamp(min=eps) 30 本）。
#:
#: MUST: max の欄を作らない。「欠けた側を f32 の最大有限値で補って clamp へ流す」は
#: 上限を持たない意味論を近似で置き換える手筋で、ADR 0017 が名指しで却下している。
CLAMP_MIN_ATTRS: AttrSchema = {
    "min": lambda value, where: _assert_finite_attr(value, where, "clamp_min の min")
}

#: torch の negative_slope。**必須で既定値補完はしない**（ADR 0015 — dec は 0.1 と 0.01 が
#: 混在し、既定に頼ると片方が黙って誤る）。torch 側の既定はハンドラが読み取って attrs に載せる。
LEAKY_RELU_ATTRS: AttrSchema = {
    "negative_slope": lambda value, where: _assert_finite_attr(
        value, where, "leaky_relu の negative_slope"
    )
}

#: ge_scalar / le_scalar / gt_scalar の比較相手。
SCALAR_COMPARE_ATTRS: AttrSchema = {
    "value": lambda value, where: _assert_finite_attr(value, where, "比較 op の value")
}

#: 負の軸表記はエクスポータ境界で正規化する規約（softmax の dim と同じ）。
CUMSUM_ATTRS: AttrSchema = {"dim": lambda value, where: _assert_integer_attr(value, where, 0)}

#: reduce 族（sum / amax / amin）の縮約軸。**宣言必須**で既定値補完はしない — 「欄が無い =
#: 最終次元」を許すと、チャネル軸の縮約を書いたつもりの IR が黙って最終次元を畳んだ別の
#: 計算として実行される（conv1d の dilation / groups と同じ理由 — ADR 0015）。
REDUCE_ATTRS: AttrSchema = {"dim": lambda value, where: _assert_integer_attr(value, where, 0)}

#: elementwise カーネルへ **params の末尾で** f32 として渡す attr（並びがそのまま params の
#: レイアウト）。エクスポータ側で持つのは、キーを跨ぐ不変条件（clamp の min <= max）を
#: shape 層で検査するため（TS 側 packages/runtime/src/ops.ts の SCALAR_PARAM_ATTRS と同義）。
SCALAR_PARAM_ATTRS: dict[str, tuple[str, ...]] = {
    "clamp": ("min", "max"),
    "clamp_min": ("min",),
    "leaky_relu": ("negative_slope",),
    "ge_scalar": ("value",),
    "le_scalar": ("value",),
    "gt_scalar": ("value",),
}

#: conv1d の attrs（ADR 0015 で dilation / groups を追加）。
#:
#: MUST: 4 つとも**宣言必須**（assert_node_contract が全キーの存在を要求する）。「欄が無い =
#: 1 固定」で担保していた「1 以外を黙って 1 で実行する経路が無い」性質は、欄を作った後は
#: **既定値補完をしない**ことだけが担保している。
CONV1D_ATTRS: AttrSchema = {
    "stride": lambda value, where: _assert_integer_attr(value, where, 1),
    "padding": lambda value, where: _assert_integer_attr(value, where, 0),
    "dilation": lambda value, where: _assert_integer_attr(value, where, 1),
    "groups": lambda value, where: _assert_integer_attr(value, where, 1),
}


def _assert_int_pair(value: Any, where: str, minimum: int, what: str) -> tuple[int, int]:
    """conv2d の空間 attr（`[H, W]` の 2 成分）。

    MUST: **長さちょうど 2 のリスト**のみ受理する。スカラ表記（torch の stride=1 が両軸に
    効く形）を併せて許すと同じ畳み込みに 2 通りの IR ができ、CSE も適合表の突合も割れる —
    正規化はエクスポータ境界（ハンドラ）の仕事。
    MUST: H と W を別のキーに割らない。2 軸 × 3 attr で 6 キーになり、「片方だけ書き忘れた
    IR」の見え方が「必須キー欠落」から「値が既定に見える」へ落ちる。
    """
    if not isinstance(value, list) or len(value) != 2:
        raise OpContractError(f"{where}: {what} は [H, W] の長さ 2 のリストでない: {value!r}")
    return (
        _assert_integer_attr(value[0], f"{where}[0]", minimum),
        _assert_integer_attr(value[1], f"{where}[1]", minimum),
    )


#: conv2d の attrs（ADR 0017）。空間 3 つは H/W の 2 成分、groups はスカラ。
#:
#: MUST: 4 つとも**宣言必須・既定値補完なし**（conv1d と同じ規律 — ADR 0015）。depthwise と
#: 非対称 stride/padding が実測に出るので、省略を許すと黙って通常畳み込み・対称パディングに
#: なる。
CONV2D_ATTRS: AttrSchema = {
    "stride": lambda value, where: _assert_int_pair(value, where, 1, "conv2d の stride"),
    "padding": lambda value, where: _assert_int_pair(value, where, 0, "conv2d の padding"),
    "dilation": lambda value, where: _assert_int_pair(value, where, 1, "conv2d の dilation"),
    "groups": lambda value, where: _assert_integer_attr(value, where, 1),
}

#: conv_transpose1d の attrs（ADR 0015）。
#:
#: MUST: stride >= 1（stride 0 はカーネルのゼロ除算・GPU ハング — recon §4）。
#: MUST: output_padding / dilation / groups の欄を作らない。実測は全て 0 / 1 / 1 で、欄を
#: 持たないことが「実測外の値を黙って既定値で実行する」経路を構造的に潰す。
CONV_TRANSPOSE1D_ATTRS: AttrSchema = {
    "stride": lambda value, where: _assert_integer_attr(value, where, 1),
    "padding": lambda value, where: _assert_integer_attr(value, where, 0),
}


def layer_norm_attrs(attrs: Mapping[str, Any], where: str) -> tuple[list[int], float]:
    """layer_norm ノードの (normalized_shape, eps)。

    検査は assert_node_contract が済ませている前提（ここは引き直すだけ）。
    """
    return (
        _assert_normalized_shape(
            attrs.get("normalized_shape"), f"{where} の attrs.normalized_shape"
        ),
        _assert_eps(attrs.get("eps"), f"{where} の attrs.eps", "layer_norm"),
    )


def rms_norm_eps(attrs: Mapping[str, Any], where: str) -> float:
    """rms_norm ノードの eps（検査は assert_node_contract が済ませている前提）。"""
    return _assert_eps(attrs.get("eps"), f"{where} の attrs.eps", "rms_norm")


def softmax_dim(attrs: Mapping[str, Any], where: str) -> int:
    """softmax ノードの縮約軸（非負の軸番号）。"""
    return _assert_integer_attr(attrs.get("dim"), f"{where} の attrs.dim", 0)


def attention_scale(attrs: Mapping[str, Any], where: str) -> float:
    """attention ノードの半スケール（検査は assert_node_contract が済ませている前提）。"""
    return _assert_finite_attr(attrs.get("scale"), f"{where} の attrs.scale", "attention の scale")


def cumsum_dim(attrs: Mapping[str, Any], where: str) -> int:
    """cumsum ノードの累積軸（非負の軸番号）。"""
    return _assert_integer_attr(attrs.get("dim"), f"{where} の attrs.dim", 0)


def reduce_dim(attrs: Mapping[str, Any], where: str) -> int:
    """reduce 族ノードの縮約軸（非負の軸番号 — 既定値補完はしない）。"""
    return _assert_integer_attr(attrs.get("dim"), f"{where} の attrs.dim", 0)


def scalar_param_values(contract: OpContract, attrs: Mapping[str, Any], where: str) -> list[float]:
    """スカラ attr を params の並び順で取り出す
    （packages/runtime/src/ops.ts の scalarParamValues と同義）。

    MUST: clamp の `min <= max` はここでしか見られない（attrs スキーマはキー単位の検査なので、
    2 つのキーに跨る不変条件を表せない）。逆転を許すと WGSL の clamp が未定義になる向きの
    IR をエクスポータが書けてしまう。
    """
    keys = SCALAR_PARAM_ATTRS.get(contract.name)
    if keys is None:
        return []
    values = [
        _assert_finite_attr(
            attrs.get(key), f"{where} の attrs.{key}", f"op '{contract.name}' の {key}"
        )
        for key in keys
    ]
    if contract.name == "clamp" and values[0] > values[1]:
        raise OpContractError(f"{where}: clamp の min {values[0]} が max {values[1]} を超える")
    return values


def conv1d_attrs(attrs: Mapping[str, Any], where: str) -> tuple[int, int, int, int]:
    """conv1d ノードの (stride, padding, dilation, groups)。"""
    return (
        _assert_integer_attr(attrs.get("stride"), f"{where} の attrs.stride", 1),
        _assert_integer_attr(attrs.get("padding"), f"{where} の attrs.padding", 0),
        _assert_integer_attr(attrs.get("dilation"), f"{where} の attrs.dilation", 1),
        _assert_integer_attr(attrs.get("groups"), f"{where} の attrs.groups", 1),
    )


def conv2d_attrs(
    attrs: Mapping[str, Any], where: str
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], int]:
    """conv2d ノードの (stride, padding, dilation, groups)。空間 3 つは (H, W) の組。"""
    return (
        _assert_int_pair(attrs.get("stride"), f"{where} の attrs.stride", 1, "conv2d の stride"),
        _assert_int_pair(attrs.get("padding"), f"{where} の attrs.padding", 0, "conv2d の padding"),
        _assert_int_pair(
            attrs.get("dilation"), f"{where} の attrs.dilation", 1, "conv2d の dilation"
        ),
        _assert_integer_attr(attrs.get("groups"), f"{where} の attrs.groups", 1),
    )


def conv_transpose1d_attrs(attrs: Mapping[str, Any], where: str) -> tuple[int, int]:
    """conv_transpose1d ノードの (stride, padding)。"""
    return (
        _assert_integer_attr(attrs.get("stride"), f"{where} の attrs.stride", 1),
        _assert_integer_attr(attrs.get("padding"), f"{where} の attrs.padding", 0),
    )


@dataclass(frozen=True)
class UniformDtypes:
    """全スロットが同じ受理集合を持ち、かつ**互いに同型**であることまで要求する契約。

    出力も入力と同型。elementwise / matmul / bmm / 行 reduce / レイアウトはこれ。
    """

    accept: frozenset[str]


@dataclass(frozen=True)
class PerSlotDtypes:
    """スロットごとに受理集合が違う契約（ADR 0012 の拡張）。

    スロット間の同型は要求せず、出力は**スロット 0（値の側）と同型**。gather の
    src=f32 / index=i32 → out=f32 が最初の例。

    MUST: 受理集合が同じ op をこちらで書かない。UniformDtypes は `mul(f32, i32)` のような
    混合を拒否する規則も担っており、per-slot に潰すとその拒否が黙って消える。
    """

    slots: tuple[frozenset[str], ...]


SlotDtypes = UniformDtypes | PerSlotDtypes


@dataclass(frozen=True)
class OpContract:
    name: str
    kind: OpKind
    #: 入力の個数（現状の op は全て単一出力）。
    arity: int
    #: スロット別の受理集合と出力導出の正本（cast だけ出力 dtype が attrs.to で決まる）。
    slot_dtypes: SlotDtypes
    #: **スロット 0 の入力 dtype → 出力 dtype**。既定は恒等で、違うのは実測に出た 3 系統だけ
    #: （比較 → bool / bool の sum → i32 / where → 値の側）。定義域はスロット 0 の受理集合と
    #: 完全一致する（_contract / _slot_contract が恒等で埋める）。
    output_dtypes: Mapping[str, str]
    #: attrs スキーマ。空なら非空 attrs は fail loudly。
    attrs: AttrSchema
    #: 入力数が可変の op（現状 `cat` のみ）。True のとき `arity` は**下限**（arity_fits）。
    #:
    #: MUST: 「アリティ検査を緩める」ために立てない。可変なのは cat の入力本数だけで、他の op は
    #: 本数そのものが契約（bias 常時ありのアリティ 3 固定など）。
    variadic: bool = False

    @property
    def dtypes(self) -> frozenset[str]:
        """受理集合の**和**（capability 突合と診断で使う射影）。

        MUST: slot_dtypes から導出する — 手書きの集合は表と乖離し、
        「対応表では実行可、契約検査で落ちる」を作る。
        """
        if isinstance(self.slot_dtypes, UniformDtypes):
            return self.slot_dtypes.accept
        return frozenset().union(*self.slot_dtypes.slots)

    def slot_accept(self, slot: int) -> frozenset[str]:
        """入力スロット `slot` の受理集合（uniform では全スロット共通）。"""
        if isinstance(self.slot_dtypes, UniformDtypes):
            return self.slot_dtypes.accept
        if not 0 <= slot < len(self.slot_dtypes.slots):
            raise OpContractError(f"op '{self.name}' に入力スロット {slot} は無い")
        return self.slot_dtypes.slots[slot]

    @property
    def attr_keys(self) -> frozenset[str]:
        """スキーマが宣言するキー（対応表突合の射影 — 二重管理しない）。"""
        return frozenset(self.attrs)

    def output_dtype(self, input_dtype: str) -> str:
        """スロット 0 の入力 dtype から出力 dtype を導く
        （packages/runtime/src/ops.ts の outputDtypeOf と同義）。
        """
        mapped = self.output_dtypes.get(input_dtype)
        if mapped is None:
            raise OpContractError(
                f"op '{self.name}' の出力 dtype 写像に入力 '{input_dtype}' が無い"
            )
        return mapped


#: op ごとの dtype 解禁表。ここに無い op は f32 のまま（一括解禁しない — ADR 0007）。
_DTYPES: dict[str, frozenset[str]] = {
    "mul": F32_I32_DTYPES,
    "sub": F32_I32_DTYPES,
    "bitwise_not": BOOL_DTYPES,
    # spline の `inside = (x >= -b) & (x <= b)`（recon §2）— bool の論理積。
    "bitwise_and": BOOL_DTYPES,
    "sum": F32_BOOL_DTYPES,
    # reshape は要素を 1 つも読み書きしない（別名化のみ）ので全語彙。実測も f32 の view と
    # i32 mask の squeeze/unsqueeze、bool マスクの unsqueeze が全て出る（recon §2）。
    RESHAPE_OP: frozenset(SEMANTIC_DTYPES),
    # gather 添字（i32）・conv 経路の bool マスクに加え、相対位置埋め込みの 4D 化（f32 —
    # recon §2 の enc_p / flow 行）。strided コピー族は dtype パラメトリックなので共用のまま。
    EXPAND_OP: frozenset(SEMANTIC_DTYPES),
    # 焼いた定数は相対位置バケット表（i32）と位置テーブル（f32）の 2 系統。bool の
    # initializer は語彙に無いので解禁しない。
    SYM_PREFIX_SLICE_OP: F32_I32_DTYPES,
}

#: 入力（スロット 0）と出力で dtype が違う op の写像。ここに無い op は恒等。
#: 比較 4 本（f32 → bool）/ bool 入力の sum（→ i32 のカウント）/ where（bool → f32）だけ。
_OUTPUT_DTYPES: dict[str, dict[str, str]] = {
    "ge": {"f32": "bool"},
    "ge_scalar": {"f32": "bool"},
    "le_scalar": {"f32": "bool"},
    "gt_scalar": {"f32": "bool"},
    "sum": {"f32": "f32", "bool": "i32"},
    WHERE_OP: {"bool": "f32"},
}


def _output_dtypes(name: str, slot_dtypes: SlotDtypes) -> dict[str, str]:
    """スロット 0 の受理集合上の写像（宣言が無い dtype は恒等で埋める）。

    MUST: 定義域をスロット 0 の受理集合と一致させる — 部分写像にすると「スロット検査は
    通ったのに出力 dtype が決まらない」穴ができる。
    """
    declared = _OUTPUT_DTYPES.get(name, {})
    domain = slot_dtypes.accept if isinstance(slot_dtypes, UniformDtypes) else slot_dtypes.slots[0]
    return {dtype: declared.get(dtype, dtype) for dtype in domain}


def _contract(
    name: str,
    kind: OpKind,
    arity: int,
    attrs: AttrSchema | None = None,
    *,
    variadic: bool = False,
) -> OpContract:
    return _slot_contract(
        name,
        kind,
        arity,
        UniformDtypes(accept=_DTYPES.get(name, F32_DTYPES)),
        attrs,
        variadic=variadic,
    )


def _slot_contract(
    name: str,
    kind: OpKind,
    arity: int,
    slot_dtypes: SlotDtypes,
    attrs: AttrSchema | None = None,
    *,
    variadic: bool = False,
) -> OpContract:
    return OpContract(
        name=name,
        kind=kind,
        arity=arity,
        slot_dtypes=slot_dtypes,
        output_dtypes=_output_dtypes(name, slot_dtypes),
        attrs=attrs if attrs is not None else {},
        variadic=variadic,
    )


#: attrs を持つ単項 op のスキーマ（無い op は attrs 空）。
_UNARY_ATTRS: dict[str, AttrSchema] = {
    "clamp": CLAMP_ATTRS,
    "clamp_min": CLAMP_MIN_ATTRS,
    "leaky_relu": LEAKY_RELU_ATTRS,
    "ge_scalar": SCALAR_COMPARE_ATTRS,
    "le_scalar": SCALAR_COMPARE_ATTRS,
    "gt_scalar": SCALAR_COMPARE_ATTRS,
}


OP_CONTRACTS: dict[str, OpContract] = {
    **{name: _contract(name, "unary", 1, _UNARY_ATTRS.get(name)) for name in UNARY_OPS},
    **{name: _contract(name, "binary", 2) for name in BINARY_OPS},
    **{name: _contract(name, "row_reduce", 1, REDUCE_ATTRS) for name in REDUCE_OPS},
    # 条件 bool と値 f32 のスロット別契約。出力は**値の側**（写像 bool → f32）。
    WHERE_OP: _slot_contract(
        WHERE_OP, "where", 3, PerSlotDtypes(slots=(BOOL_DTYPES, F32_DTYPES, F32_DTYPES))
    ),
    CUMSUM_OP: _contract(CUMSUM_OP, "cumsum", 1, CUMSUM_ATTRS),
    MATMUL_OP: _contract(MATMUL_OP, "matmul", 2),
    BMM_OP: _contract(BMM_OP, "bmm", 2),
    # 最初のスロット別 dtype 契約: 値 f32 と添字 i32 が混在し、出力は値の側と同型。
    GATHER_OP: _slot_contract(
        GATHER_OP, "gather", 2, PerSlotDtypes(slots=(F32_DTYPES, I32_DTYPES))
    ),
    CAST_OP: _slot_contract(
        CAST_OP, "cast", 1, UniformDtypes(accept=frozenset(SEMANTIC_DTYPES)), CAST_ATTRS
    ),
    RESHAPE_OP: _contract(RESHAPE_OP, "reshape", 1),
    PERMUTE_OP: _contract(PERMUTE_OP, "permute", 1, PERMUTE_ATTRS),
    EXPAND_OP: _contract(EXPAND_OP, "expand", 1),
    # レイアウト第 2 群（ADR 0014）。実測は全て f32（enc_p の m_p/logs_p 分割・coupling の
    # 96/96 分割と cat・相対位置 value 側の pad・flow/sdp の Flip — recon §2）。
    SLICE_OP: _contract(SLICE_OP, "slice", 1, SLICE_ATTRS),
    # 唯一の可変アリティ op。arity は**下限** 2（1 本の cat は恒等コピーで実測にも出ない）。
    CAT_OP: _contract(CAT_OP, "cat", 2, AXIS_ATTRS, variadic=True),
    PAD_OP: _contract(PAD_OP, "pad", 1, PAD_ATTRS),
    FLIP_OP: _contract(FLIP_OP, "flip", 1, AXIS_ATTRS),
    SYM_PREFIX_SLICE_OP: _contract(
        SYM_PREFIX_SLICE_OP, "sym_prefix_slice", 1, SYM_PREFIX_SLICE_ATTRS
    ),
    # 融合 op（ADR 0012）。bias / affine を持つ 3 本はアリティ 3 固定 — 実測が bias 常時
    # ありで、「bias 無し」を表す欄を作らないことがそのまま fail loudly になる。
    LINEAR_OP: _contract(LINEAR_OP, "linear", 3),
    LAYER_NORM_OP: _contract(LAYER_NORM_OP, "layer_norm", 3, LAYER_NORM_ATTRS),
    # bias が無いのでアリティ 2（ADR 0017）。weight 無しの形はハンドラが ones 合成で
    # アリティ 2 へ正規化する — ゼロ bias 合成（ADR 0015）と同じ手筋。
    RMS_NORM_OP: _contract(RMS_NORM_OP, "rms_norm", 2, RMS_NORM_ATTRS),
    SOFTMAX_OP: _contract(SOFTMAX_OP, "softmax", 1, SOFTMAX_ATTRS),
    # 融合 attention（ADR 0023）。q / k / v の 3 本とも f32 で同型（uniform 契約）。
    ATTENTION_OP: _contract(ATTENTION_OP, "attention", 3, ATTENTION_ATTRS),
    # 値 f32 と添字 i32 のスロット別契約（gather と同型 — 出力は値の側と同型）。
    EMBEDDING_OP: _slot_contract(
        EMBEDDING_OP,
        "embedding",
        2,
        PerSlotDtypes(slots=(F32_DTYPES, I32_DTYPES)),
        EMBEDDING_ATTRS,
    ),
    # 値 f32 と条件 bool のスロット別契約。出力はスロット 0（値の側）と同型。
    MASKED_FILL_OP: _slot_contract(
        MASKED_FILL_OP,
        "masked_fill",
        2,
        PerSlotDtypes(slots=(F32_DTYPES, BOOL_DTYPES)),
        MASKED_FILL_ATTRS,
    ),
    CONV1D_OP: _contract(CONV1D_OP, "conv1d", 3, CONV1D_ATTRS),
    CONV2D_OP: _contract(CONV2D_OP, "conv2d", 3, CONV2D_ATTRS),
    # bias 無し conv はエクスポータのゼロ bias 合成でアリティ 3 に正規化される（ADR 0015）—
    # カーネルにも契約にも arity 分岐を持ち込まない。
    CONV_TRANSPOSE1D_OP: _contract(
        CONV_TRANSPOSE1D_OP, "conv_transpose1d", 3, CONV_TRANSPOSE1D_ATTRS
    ),
}

#: convert が emit しうる IR op 名の正本。語彙の増加を明示行為にするための門。
EMITTABLE_OPS = frozenset(OP_CONTRACTS)


#: rank 上限が効く op（strided コピー族 — カーネルの params は rank 固定）。
#: MUST: TS 側（packages/runtime/src/ops.ts の computeOutputShape）と同じ集合。片方だけに置くと
#: 「export は緑、ブラウザだけ落ちる」が rank 軸で復活する。
STRIDED_RANK_OPS = frozenset(
    {PERMUTE_OP, EXPAND_OP, SLICE_OP, CAT_OP, SYM_PREFIX_SLICE_OP, MASKED_FILL_OP}
)


def assert_strided_rank(rank: int, what: str, where: str) -> None:
    """strided コピー族が実行できる rank
    （packages/runtime/src/ops.ts の assertStridedRank と同義）。

    MUST: 契約層で見る。超過を codegen まで落とすと「契約検査は通ったのに実行段で内部
    エラー」になり、どの op のどの値が悪いのか診断に出ない。
    """
    if not 1 <= rank <= STRIDED_RANK:
        raise OpContractError(
            f"{where}: {what} の rank {rank} は 1..{STRIDED_RANK} の外（strided カーネルの上限）"
        )


def arity_fits(contract: OpContract, count: int) -> bool:
    """入力の本数が契約に合うか（可変アリティ op では `arity` を**下限**として読む）。

    MUST: 判定を呼び出し側に散らさない（packages/runtime/src/ops.ts の arityFits と同義）。
    「固定なら ==、可変なら >=」を契約検査と shape 計算の 2 箇所に書くと、片方だけ古い判定が残る。
    """
    return count >= contract.arity if contract.variadic else count == contract.arity


def describe_arity(contract: OpContract) -> str:
    """契約のアリティの表示形（可変なら「N 本以上」）。"""
    return f"{contract.arity} 本以上" if contract.variadic else str(contract.arity)


def resolve_op_contract(op: str) -> OpContract:
    contract = OP_CONTRACTS.get(op)
    if contract is None:
        raise OpContractError(f"op '{op}' は契約表に無い")
    return contract


def cast_target_dtype(attrs: Mapping[str, Any], where: str) -> str:
    """cast ノードの変換先 dtype（attrs の検査は assert_node_contract が済ませている前提）。"""
    return _assert_cast_target(attrs.get("to"), f"{where} の attrs.to")


def assert_node_contract(node: IrNode, where: str) -> OpContract:
    """ノードが契約に適合することを検査して契約を返す（shape は含まない）。"""
    contract = resolve_op_contract(node.op)
    if not arity_fits(contract, len(node.ins)):
        raise OpContractError(
            f"{where}: op '{node.op}' の入力数が {len(node.ins)}"
            f"（契約は {describe_arity(contract)}）"
        )
    if len(node.outs) != 1:
        raise OpContractError(
            f"{where}: op '{node.op}' の出力数が {len(node.outs)}（現状の op は全て単一出力）"
        )
    unknown = sorted(key for key in node.attrs if key not in contract.attrs)
    if unknown:
        raise OpContractError(f"{where}: op '{node.op}' の契約外 attrs {unknown}")
    for key, check in contract.attrs.items():
        if key not in node.attrs:
            raise OpContractError(f"{where}: op '{node.op}' の必須 attr '{key}' が無い")
        check(node.attrs[key], f"{where} の attrs.{key}")
    return contract


def resolve_node_dtypes(
    contract: OpContract,
    node: IrNode,
    input_dtypes: list[str],
    declared_output: str,
    where: str,
) -> str:
    """ノードの意味論 dtype を検査して出力 dtype を返す（packages/runtime/src/ops.ts と同義）。

    MUST: 出力 dtype は宣言を鵜呑みにせず契約から導く。導出は 2 通りだけ — cast は attrs.to、
    それ以外は**スロット 0 の dtype を契約の写像に通す**（既定は恒等）。uniform 契約では加えて
    スロット間の同型も要求する。宣言と食い違ったグラフはランタイム側で別 TypedArray として
    読まれる沈黙誤値になる。
    """
    for slot, (name, dtype) in enumerate(zip(node.ins, input_dtypes, strict=True)):
        accept = contract.slot_accept(slot)
        if dtype not in accept:
            uniform = isinstance(contract.slot_dtypes, UniformDtypes)
            slot_note = "" if uniform else f"スロット {slot} は"
            raise OpContractError(
                f"{where} の入力 '{name}': op '{contract.name}' は{slot_note}"
                f"意味論 dtype '{dtype}' を実行できない（対応: {', '.join(sorted(accept))}）"
            )
    if contract.kind == "cast":
        expected = cast_target_dtype(node.attrs, where)
    else:
        # スロットごとに受理集合が違う op は、スロット間の同型を要求しない（それが per-slot の
        # 意味）。uniform 契約だけが混在を拒否する。
        if isinstance(contract.slot_dtypes, UniformDtypes) and any(
            dtype != input_dtypes[0] for dtype in input_dtypes
        ):
            raise OpContractError(
                f"{where}: op '{contract.name}' の入力 dtype が混在（{', '.join(input_dtypes)}）"
            )
        expected = contract.output_dtype(input_dtypes[0])
    if declared_output != expected:
        raise OpContractError(
            f"{where}: 出力 '{node.outs[0]}' の宣言 dtype '{declared_output}' が"
            f" 契約の '{expected}' と違う"
        )
    return expected
