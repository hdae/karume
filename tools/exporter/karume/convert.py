"""torch.export 済み ExportedProgram → IR v1 グラフ（docs/ir-v1.md）。

入力はキュレーション済み decomp table を通した ExportedProgram（functionalize 済み・
保存 op は高位のまま）。処理は 3 段:

1. 定数畳み込み: 定数と shape シンボルだけに依存する部分木を、各シンボルの**最大値
   （Tmax）で実評価**して initializer に焼き、実行時は記号 prefix スライス
   （`sym_prefix_slice`）で先頭を切り出す（ADR 0010）。DeBERTa の相対位置バケット表
   （log / ceil を含む）を export 時に焼き込むのが主目的 — バケット境界の 1ulp 差 →
   gather 添字 1 ずれのバグクラスを構造的に排除する。
2. aten → IR op 対応表による変換。未対応 aten op は **全件列挙して** fail loudly
   （1 件ずつ落とすと語彙を埋める側が何本足りないのか分からない — ADR 0005 の診断規律）。
3. 純 op の CSE（同一 (op, ins, attrs, shape) を 1 本に畳む）。

MUST: 畳み込み allowlist（{@link FOLDABLE_OPS}）は「prefix スライスと可換な op」だけ —
`f(T)[0:T,…] == f(Tmax)[0:T,…]` が成立しない op（次元縮約、T を値として使う正規化等）を
入れてはならない。ただし**この不変条件は allowlist だけでは守れない**（allowlist 掲載 op と
sym_size の合成で T が「値」として部分木に入りうる）ため、2 段で守る:

1. シンボルの消費位置を見る — extent（長さ・形状）位置だけを {@link SYMBOL_EXTENT_ARGS} で
   許し、値位置へ届いた部分木は foldable から外す（symbol-as-data の拒否）。外れた op は
   通常の lowering へ落ち、語彙外なら未対応 op として export ごと拒否される。
2. 畳み込みのたびに Tmax と別の点で**2 点評価して prefix 一致を実測する**
   （_check_prefix_commutes）。

2 点評価は防波堤であって証明ではない — `(scalar_tensor(T)−Tmax)(scalar_tensor(T)−(Tmax−1))`
のように 2 点で偶然一致する反例があり、単独では silent wrong value を通す。1 が本体。

シンボルは複数持てる。IR 上の名前は呼び出し側が symbol_names で与え、user 入力
placeholder の出現順（= forward の引数順）で割り当てる — torch.export の内部シンボル名
（`s97` 等）はグラフごとに変わるので IR に持ち込まない。1 つの次元に現れるシンボルは
1 つだけ（`s20+s97` 形は fail loudly）。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, NamedTuple

import torch

# MUST: `torch.ops.torchvision.*` は torchvision の C++ ライブラリを読み込まないと引けない
# （ハンドラのキーが書けない）。条件付き import で soft 依存にすると「deform_conv2d を
# 未対応 op として落とす env」と「通る env」が分岐するので、基本依存として素で import する
# （ADR 0055 決定 7）。
import torchvision  # noqa: F401  -- torch.ops.torchvision の登録が副作用
from torch.export import ExportedProgram
from torch.export.graph_signature import OutputKind
from torch.fx import Node

from karume.dims import MAX_SAFE_INT, DimExpr, format_dim, is_symbol_name
from karume.extents import extent_key, same_extents
from karume.ir import (
    IrGraph,
    IrInitializer,
    IrInput,
    IrNode,
    IrStorage,
    IrValue,
)
from karume.normalize import SAFE_SOFTMAX_META
from karume.ops import EMITTABLE_OPS, STRIDED_RANK
from karume.shapes import assert_graph_shapes

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

#: i32 の値域（境界正規化の検査に使う）。
I32_MIN = -(2**31)
I32_MAX = 2**31 - 1

#: 焼いた定数の torch dtype → IR の格納 dtype（docs/ir-v1.md「値と型」）。
#:
#: f32 は「f32 値の符号化」語彙の素通し、`i32` は生の int32（ADR 0010 の明示的な例外）。
#: i64 はここに来る前に normalize_boundary_tensor が i32 へ落とす。ここに無い dtype
#: （bool 等）の定数は initializer の語彙が無いので fail loudly。
_CONST_STORAGE = {torch.float32: "f32", torch.int32: "i32"}


def normalize_boundary_tensor(tensor: torch.Tensor, where: str) -> torch.Tensor:
    """境界テンソル（入出力の実データ）を IR v1 の意味論 dtype の実表現へ落とす。

    - i64 → i32（**値域外は fail loudly** — 黙って切り詰めると添字が静かにずれる）
    - bool → u32 の 0 / 1（GPU 格納と同じ規約 — ADR 0009）
    - f32 / i32 はそのまま

    MUST: 変換点はここ 1 箇所。呼び出し側で `.to(torch.int32)` を書くと値域検査が
    経路ごとに抜ける。
    """
    dtype = tensor.dtype
    if dtype is torch.float32 or dtype is torch.int32:
        return tensor.contiguous()
    if dtype is torch.bool:
        return tensor.to(torch.uint32).contiguous()
    if dtype is torch.int64:
        if tensor.numel() > 0:
            low = int(tensor.min())
            high = int(tensor.max())
            if low < I32_MIN or high > I32_MAX:
                raise ValueError(
                    f"{where}: i64 の値域 [{low}, {high}] が i32 に収まらない"
                    f"（[{I32_MIN}, {I32_MAX}]）— IR v1 は i64 を持たない（ADR 0009）"
                )
        return tensor.to(torch.int32).contiguous()
    raise NotImplementedError(
        f"{where}: torch dtype {dtype} は IR v1 の意味論 dtype に落とせない"
        f"（{', '.join(sorted(set(DTYPE_NAMES.values())))}）"
    )


def _check_clone(node: Node) -> None:
    """aten.clone → 恒等（IR に出さず入力の別名にする）。

    根拠: IR v1 の値は**常に連続**（permute / expand は strided コピーで連続化して出し、
    reshape は要素順を変えない — ADR 0011）。連続な値の contiguous 化コピーは恒等なので、
    実測 24 本（recon §2 の① permute 直後の材料化 ②③ dropout / functionalization 由来）は
    まとめてここで吸収できる。

    MUST: `memory_format` が連続以外の形は落とす。channels_last 等は「連続化」ではなく
    別レイアウトへの並べ替えで、恒等に潰すと要素順が黙ってずれる。
    """
    extra = sorted(set(node.kwargs) - {"memory_format"})
    _expect(not extra, node, f"kwargs {extra} を伴う clone は未対応")
    memory_format = node.kwargs.get("memory_format", torch.contiguous_format)
    _expect(
        memory_format in (torch.contiguous_format, torch.preserve_format),
        node,
        f"memory_format={memory_format} の clone は未対応（IR の値は常に連続 — ADR 0011）",
    )


#: IR に出さない no-op（値は入力の別名になる）。値は「受理条件の検査関数」で、
#: 無条件に別名化してよい op は None。
SKIP_OPS: dict[Any, Callable[[Node], None] | None] = {
    aten.alias.default: None,
    aten._assert_tensor_metadata.default: None,
    aten.clone.default: _check_clone,
}

#: 畳み込みの 2 点評価点。MAX は全シンボルをその上限（Tmax）に、PROBE は全シンボルを
#: **Tmax − 1** に置いた点。可換性の検査は両点を同時に切り替えて比べる。
#:
#: MUST: 第 2 点は Tmax と別の値で、かつ **2 以上**（torch.export の 0/1 特殊化は
#: `Dim(min=2)` 運用の前提で、1 以下の点は「その長さでだけ成り立つ特殊形」を評価しうる）。
#: 同一点になると 2 点評価が恒真化して検査が黙って無意味になる — _sym_range が門になる。
FOLD_POINT_MAX = "max"
FOLD_POINT_PROBE = "probe"

#: 2 点評価の第 2 点として許す最小値（0/1 特殊化の回避線）。
MIN_PROBE_VALUE = 2

#: 定数畳み込みの allowlist。
#:
#: 各 op が prefix スライスと可換な理由:
#:
#: - `arange(T)` は `arange(Tmax)` の先頭 T 要素
#: - elementwise / 比較 / 選択（where）は要素ごと（broadcast 込み — 対応次元が保たれる）
#: - slice / squeeze / unsqueeze / view / permute は添字の付け替えのみ
#: - `full` / `scalar_tensor` / `clone` / `_to_copy` は値そのものに触らない
#:
#: 可換でない使われ方のうち「シンボルを値として消費する形」は {@link SYMBOL_EXTENT_ARGS} が
#: foldable から外し、残りは _check_prefix_commutes が 2 点評価で落とす — **宣言ではなく
#: 検査が担保する**。除外の理由は 3 つ: ① 実体化が爆発する op（expand。frontier で止める —
#: ただし no-op expand だけは例外で `_classify_foldable` が通す）② 非決定な op（RNG）
#: ③ 縮約・反転など prefix と可換でないことが自明な op。
#: 語彙の増加は明示行為 — 表に無い op は畳まず、消費されれば未対応 op として落ちる。
FOLDABLE_OPS = {
    aten.arange.default,
    aten.arange.start_step,
    aten.full.default,
    aten.scalar_tensor.default,
    aten.clone.default,
    aten._to_copy.default,
    aten.add.Tensor,
    aten.sub.Tensor,
    aten.mul.Tensor,
    aten.div.Tensor,
    aten.neg.default,
    aten.abs.default,
    aten.exp.default,
    aten.log.default,
    aten.sqrt.default,
    aten.cat.default,
    aten.permute.default,
    aten.slice.Tensor,
    aten.squeeze.dim,
    aten.squeeze.dims,
    aten.unsqueeze.default,
    aten.view.default,
    # 相対位置バケット表（recon §2 の foldable 群 — ADR 0010）。いずれも要素ごとの写像で、
    # 行 t / 列 j の値が T に依らないため prefix と可換。
    aten.sign.default,
    aten.ceil.default,
    aten.clamp.default,
    aten.lt.Scalar,
    aten.gt.Scalar,
    aten.le.Scalar,
    aten.bitwise_and.Tensor,
    aten.where.self,
    # RoPE と定数生成（ADR 0016 の FOLDABLE 拡張）。**実測で畳み frontier に現れた op だけ**を
    # 足す（ADR の候補表のうち add.Scalar / mul.Scalar だけは正規化パスが先に潰すため
    # frontier に現れず、足していない）。
    #
    # - `sin` / `cos` — RoPE の位置表そのもの。定数のうちに潰せば実行時の 1 dispatch が丸ごと
    #   消えるので、`sin` が IR 語彙に入った後もここから外さない（`cos` は語彙に無いため、
    #   畳めない = 落ちるのは従来どおり）。実行時値を取る `sin` だけが `_simple` へ落ちる
    # - `reciprocal` / `pow.Scalar` — inv_freq の生成式（`1 / theta**(i/D)`）
    # - `repeat` — DiT の rope 表を patch グリッドへ配る段（葉は定数のみ・出力は 1024×128 級）
    # - `bmm` — conditioner / Qwen3 の `inv_freq[…] @ position_ids[…]`（[1,D/2,1] × [1,1,T]）
    aten.sin.default,
    aten.cos.default,
    aten.reciprocal.default,
    aten.pow.Scalar,
    aten.repeat.default,
    aten.bmm.default,
    # 因果マスクの生成（transformers の `create_causal_mask` — 葉は arange と定数だけ）。
    # ここを畳まないと i64 の添字計算がそのまま IR に残る（ADR 0009 の境界正規化の射程外）。
    aten.cumsum.default,
    aten.index.Tensor,
    aten.eq.Tensor,
    aten.le.Tensor,
    aten.ne.Scalar,
    # sliding-window と因果マスクの合成（Gemma3 の `create_masks_for_generate` — 2 本の
    # 帯マスクを `|` で重ね、窓の外を `>` で判定する）。どちらも要素ごとの比較 / 論理和で、
    # 行 t / 列 j の値が T に依らないため prefix と可換（2 点評価も通ることを実測）。
    aten.bitwise_or.Tensor,
    aten.gt.Tensor,
    # 双方向（非因果）マスクの生成（transformers の `bidirectional_mask_function` は
    # `q_idx >= 0` を全域 true の種として使う）。比較 1 本で、行 t / 列 j の値が T に
    # 依らないため prefix と可換。
    aten.ge.Scalar,
}

#: 畳み込み対象 op で **シンボルが現れてよい引数**（= extent = 長さ・形状）の名前。
#:
#: MUST: ここに載らない引数へシンボルが届いたノードは、シンボルを**値**として消費した
#: （テンソルデータへ昇格させた）ものとして foldable から外す。extent 位置なら T が動かすのは
#: 焼いた定数の**長さ**だけで prefix 同型が保たれるが、値位置に入ると要素の**中身**が Tmax
#: でしか正しくない定数になり、2 点評価は偶然一致しうる（`scalar_tensor(T)` から作る
#: `(T−Tmax)(T−(Tmax−1))` が反例 — 焼けば T=Tmax, Tmax−1 以外で黙って値が変わる）。
#:
#: - `arange` — `end` だけ。値は start / step が決め、T は長さしか動かさない
#: - `full` / `view` / `expand` / `repeat` — 形（size / repeats）は extent、`fill_value` は値
#: - `slice` — `end` だけ（`start` が T 依存だと先頭以外を切り出す = prefix と可換でない）
#:
#: 表に無い op（`scalar_tensor` / 二項 elementwise / スカラ比較 / 軸指定の dim 等）は extent
#: 引数を持たない — シンボルが届いた時点で値としての消費。分類が不明な位置は除外側へ倒す。
SYMBOL_EXTENT_ARGS: dict[Any, frozenset[str]] = {
    aten.arange.default: frozenset({"end"}),
    aten.arange.start_step: frozenset({"end"}),
    aten.full.default: frozenset({"size"}),
    aten.view.default: frozenset({"size"}),
    aten.expand.default: frozenset({"size"}),
    aten.repeat.default: frozenset({"repeats"}),
    aten.slice.Tensor: frozenset({"end"}),
}

#: 分解を止める（融合を保つ）高位 op = **ADR 0007 の 9 op**（ADR 0012 で gelu のみから拡張）。
#:
#: 分解させるとカーネル粒度と 1 対 1 でなくなるものだけを外す:
#:
#: - `linear` — addmm + view に散る（実測 16 本 / 2 層）。融合カーネル 1 本が正。
#: - `layer_norm` — native_layer_norm（**3 出力**）+ getitem になり、IR v1 の単一出力前提と
#:   衝突する。ここは保存が事実上必須。
#: - `softmax` — `aten._softmax` になるだけで得が無い。safe-softmax の 1 カーネルを保つ。
#: - `gelu` — erf / tanh 近似の合成に散る（M0 から保存）。
#: - `conv1d` / `conv2d` / `conv_transpose1d` — 汎用 `aten.convolution` 形になる。
#: - `embedding` — core 分解では元々分解されないが、保存リストに載せて意図を明示する。
#: - `masked_fill` — where + scalar_tensor に散る。埋め値を attrs に載せる形を保つ。
#: - `leaky_relu` — ADR 0015 で追加（9 op → 10 op）。分解形は `gt_scalar + mul + where` で
#:   中間バッファが 1.5〜2 倍に膨らみ、dec のメモリ見積の前提が崩れる。
#:
#: - `rms_norm` — ADR 0017 で追加（10 op → 11 op）。diffusers `nn.RMSNorm` 由来の
#:   `aten.rms_norm` を保存する経路で、手書き分解形（Qwen3 / DiT）は保存では畳めないので
#:   `normalize._fold_rms_norm` が受け持つ（供給ルート 2 系統 — ADR 0017）。
#:
#: NOTE: conv_transpose1d は ADR 0015 で、conv2d は ADR 0017 でカーネル・契約表・ハンドラが
#: 揃った（それぞれ dec の ups 5 本 / Anima VAE decoder の 37 本）。
PRESERVED_OP_PREFIXES = (
    "aten.conv1d.",
    "aten.conv2d.",
    "aten.conv_transpose1d.",
    "aten.embedding.",
    "aten.gelu.",
    "aten.layer_norm.",
    "aten.leaky_relu.",
    "aten.linear.",
    "aten.masked_fill.",
    "aten.rms_norm.",
    "aten.softmax.",
)

#: 融合 attention（ADR 0023）を保存する接頭辞。
#:
#: MUST: 既定の {@link PRESERVED_OP_PREFIXES} には**入れない**。表はグローバルなので、足すと
#: 契約外のマスク（bool のまま kwargs で渡る形・`[B,1,M,N]` 等）を持つ SDPA まで保存対象に
#: なり、`_h_attention` の fail loudly で **export できなくなる**。有効化は
#: `curated_decompositions(preserved=…)` を通した**ターゲット別**の選択で行う
#: （ADR 0016 の safe-softmax ガード除去パスをそのまま効かせ続けるため）。
ATTENTION_OP_PREFIX = "aten.scaled_dot_product_attention."

#: SDPA を保存する preserved 集合（マスク無し、または加算型 `[1,1,M,N]` マスクだけを持つ
#: ターゲット専用 — DiT / VAE decoder / EmbeddingGemma）。
PRESERVED_OP_PREFIXES_WITH_ATTENTION = (*PRESERVED_OP_PREFIXES, ATTENTION_OP_PREFIX)


class UnsupportedAtenOpsError(NotImplementedError):
    """IR 語彙に対応表を持たない aten op（全件列挙）。"""

    def __init__(self, ops: dict[str, list[str]]) -> None:
        self.ops = dict(ops)
        listed = "; ".join(
            f"{target} (nodes: {', '.join(names)})" for target, names in sorted(ops.items())
        )
        super().__init__(f"未対応 aten op {len(ops)} 種: {listed}")


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


def _carries_symbol(value: Any) -> bool:
    """引数が **シンボルを値として運ぶ**か（SymInt そのもの / SymInt を運ぶノード）。

    テンソル引数は False — テンソル経由の伝播は「消費したノードを foldable から外す」形で
    部分木単位に効くので、ここで見るのは記号が新たに入ってくる口だけ。
    """
    if isinstance(value, Node):
        val = value.meta.get("val")
        return not isinstance(val, torch.Tensor) and _has_free_symbols(val)
    if isinstance(value, (list, tuple)):
        return any(_carries_symbol(item) for item in value)
    if isinstance(value, torch.Tensor):
        return False
    return _has_free_symbols(value)


def _uses_symbol_as_data(node: Node) -> bool:
    """シンボルを extent 以外の引数位置（= 値）で消費するノードか（{@link SYMBOL_EXTENT_ARGS}）。

    引数名は op のスキーマから引く — 位置引数と kwargs のどちらで来ても同じ判定にするため。
    """
    allowed = SYMBOL_EXTENT_ARGS.get(node.target, frozenset())
    names = [argument.name for argument in node.target._schema.arguments]
    named = [*zip(names, node.args, strict=False), *node.kwargs.items()]
    return any(name not in allowed and _carries_symbol(value) for name, value in named)


def _bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """2 つのテンソルを dtype / shape 込みで **ビット厳密**に比べる。

    MUST: 畳み込みの 2 点比較を数値等価（`torch.equal`）でやると `-0.0 == 0.0` が真になり、
    符号ビットだけ違う定数を焼く形が素通りする（initializer は signed zero をそのまま運ぶので、
    実行時に符号が黙って変わる）。浮動小数は連続バイト列の整数 view で比べ、NaN も payload
    まで一致した時だけ通す。
    """
    if left.dtype is not right.dtype or left.shape != right.shape:
        return False
    if left.is_floating_point():
        left = left.contiguous().flatten().view(torch.uint8)
        right = right.contiguous().flatten().view(torch.uint8)
    return torch.equal(left, right)


def _holds_node(value: Any) -> bool:
    if isinstance(value, Node):
        return True
    if isinstance(value, (list, tuple)):
        return any(_holds_node(item) for item in value)
    return False


class Converter:
    def __init__(self, ep: ExportedProgram, *, symbol_names: Sequence[str] = ("T",)) -> None:
        self.ep = ep
        self.symbol_names = tuple(symbol_names)
        if len(set(self.symbol_names)) != len(self.symbol_names):
            raise ValueError(f"symbol_names に重複がある: {self.symbol_names}")
        for name in self.symbol_names:
            # 名前が次元表記の文法に載らないと、書けてもランタイムが読めない次元になる。
            if not is_symbol_name(name):
                raise ValueError(f"IR シンボル名 {name!r} は次元表記の文法に載らない")
        self.gm = ep.graph_module
        signature = ep.graph_signature
        self.placeholder_fqn: dict[str, str] = {
            **signature.inputs_to_parameters,
            **signature.inputs_to_buffers,
            **signature.inputs_to_lifted_tensor_constants,
        }
        self.user_inputs = set(signature.user_inputs)
        self.lifted_constants = set(signature.inputs_to_lifted_tensor_constants)
        self.state: dict[str, torch.Tensor] = {
            **dict(ep.named_parameters()),
            **dict(ep.named_buffers()),
            **ep.constants,
        }
        self.graph = IrGraph()
        self.tensors: dict[str, torch.Tensor] = {}
        # FX ノード名 → IR 値名
        self.value_of: dict[str, str] = {}
        # SKIP op の遅延エイリアス
        self.alias_of: dict[str, Node] = {}
        # torch 内部シンボル名 → IR シンボル名
        self.sym_names: dict[str, str] = {}
        # IR シンボル名 → (下限, 上限)。上限が Tmax（畳み込みの第 1 評価点）。
        #
        # MUST: 出どころは ExportedProgram の range_constraints 1 箇所（= dynamic_shapes の
        # `Dim(min=…, max=…)` が torch に登録した制約そのもの）。呼び出し側から Tmax を
        # 別引数で受けると dynamic_shapes と二重管理になり、食い違ったときに「宣言より
        # 短い定数を焼いて実行時に範囲外」が黙って通る。
        self.sym_ranges: dict[str, tuple[Any, Any]] = {}
        # MUST: キーは**素のシンボル**だけを拾う。派生次元（`Dim("S") + 1519` を入力 shape に
        # 宣言した形 — ADR 0046）を dynamic_shapes に書くと range_constraints に sympy の
        # `Add`（`s27 + 1519`）がキーとして並び、`.name` を持たないので走査ごと落ちる。
        # 派生次元の値域は素のシンボルの値域から決まる冗長情報なので、除いて失うものは無い。
        self._internal_ranges: dict[str, tuple[Any, Any]] = {
            symbol.name: (value_range.lower, value_range.upper)
            for symbol, value_range in ep.range_constraints.items()
            # sympy を直接 import しない（torch の推移依存で、宣言依存には無い）— 素の
            # シンボルであることは sympy 自身が全式に持たせる述語で見る。
            if getattr(symbol, "is_Symbol", False)
        }
        # 命名済みフラグ（_assign_input_symbols が立てる — 以後の新規命名は拒否）
        self.symbols_assigned = False
        # 評価点 → (ノード名 → 評価結果)。2 点評価するので点ごとに分ける。
        self._fold_cache: dict[str, dict[str, Any]] = {}
        self._const_by_digest: dict[str, str] = {}
        self._cse: dict[str, str] = {}

    # ---- シンボル -------------------------------------------------------

    def _sym_parts(self, value: torch.SymInt) -> DimExpr:
        """SymInt を一次式 coeff·sym+offset に分解する。

        受理は係数 >= 1・オフセット >= 0 の整数のみ。二次式と、1 つの次元に 2 つ以上の
        シンボルが混ざる式（`s20+s97`）は正準文法に載らないので fail loudly。
        """
        expr = value.node.expr
        symbols = list(expr.free_symbols)
        if len(symbols) != 1:
            reason = "1 次元に複数シンボルが混ざる" if len(symbols) > 1 else "解釈できない"
            raise NotImplementedError(f"{reason} shape 式は IR v1 非対応: {expr}")
        symbol = symbols[0]
        coeff = expr.coeff(symbol, 1)
        offset = expr.coeff(symbol, 0)
        if (
            not getattr(coeff, "is_Integer", False)
            or not getattr(offset, "is_Integer", False)
            or coeff * symbol + offset != expr
            or not 1 <= int(coeff) <= MAX_SAFE_INT
            or not 0 <= int(offset) <= MAX_SAFE_INT
        ):
            raise NotImplementedError(f"shape 式は IR v1 非対応: {expr}")
        return DimExpr(coeff=int(coeff), sym=self._sym_name(symbol.name), offset=int(offset))

    def _sym_name(self, internal: str) -> str:
        """torch 内部シンボルの IR 名を引く（命名は _assign_input_symbols だけが行う）。"""
        name = self.sym_names.get(internal)
        if name is not None:
            return name
        if self.symbols_assigned:
            # 入力 shape に現れないシンボルはランタイムが束縛できない（束縛は入力 shape の
            # 次元位置からしか取らない — docs/ir-v1.md）。IR に載せれば実行不能になる。
            raise NotImplementedError(
                f"入力 shape に現れないシンボル {internal} が値に現れた"
                f"（入力由来={self.sym_names}）— ランタイムが束縛できないため IR に載せられない"
            )
        if len(self.sym_names) >= len(self.symbol_names):
            raise NotImplementedError(
                f"IR シンボル名が足りない: 内部シンボル {internal} に割り当てる名前が無い"
                f"（symbol_names={list(self.symbol_names)}、割り当て済み={self.sym_names}）"
                " — convert(symbol_names=...) に名前を追加すること"
            )
        name = self.symbol_names[len(self.sym_names)]
        self.sym_names[internal] = name
        # 上限は畳み込みの評価点でしか要らないので、ここでは登録だけして検査しない
        # （記号依存の定数を持たないグラフは Dim(max=…) 無しでも export できる）。
        found = self._internal_ranges.get(internal)
        if found is not None:
            self.sym_ranges[name] = found
        return name

    def _sym_range(self, name: str) -> tuple[int, int]:
        """シンボルの (下限, Tmax)。2 点評価に使えない範囲は fail loudly。

        MUST: Tmax は有限でなければならない（`Dim(max=…)` 未設定は `int_oo`）。無限のまま
        既定値で進めると、焼く定数の長さを勝手に決めることになる。
        MUST: 第 2 評価点 `Tmax − 1` が下限以上かつ {@link MIN_PROBE_VALUE} 以上であること。
        満たさないと 2 点評価が同一点に潰れて可換性検査が恒真化する（= 検査が黙って消える）。
        """
        found = self.sym_ranges.get(name)
        if found is None:
            raise NotImplementedError(
                f"シンボル {name} の値域が ExportedProgram の range_constraints に無い"
                " — 記号依存の定数を畳むには dynamic_shapes の Dim に上限が要る"
            )
        raw_lower, raw_upper = found
        # MUST: 有限性は float 経由で見る。`Dim(max=…)` 未設定の上限は torch の `int_oo` で、
        # sympy の `is_finite` は True を返す一方 `int()` は内部で例外になる（型で判定できない）。
        if not math.isfinite(float(raw_upper)):
            raise NotImplementedError(
                f"シンボル {name} の上限が有限でない（{raw_upper}）— 記号依存の定数は Tmax で"
                f" 焼く必要がある（dynamic_shapes を Dim('…', min=…, max=Tmax) で宣言すること）"
            )
        upper = int(raw_upper)
        lower = max(int(raw_lower), 0)
        if upper - 1 < max(lower, MIN_PROBE_VALUE):
            raise ValueError(
                f"シンボル {name} の値域 [{lower}, {upper}] では畳み込みの 2 点評価ができない"
                f"（第 2 点 {upper - 1} が下限 {lower} / 0-1 特殊化の回避線 {MIN_PROBE_VALUE}"
                " を下回り、可換性検査が恒真化する）"
            )
        return lower, upper

    def _point_value(self, name: str, point: str) -> int:
        """評価点でのシンボル値。PROBE は Tmax − 1（多シンボルでは全部を同時に動かす）。"""
        _, upper = self._sym_range(name)
        return upper if point == FOLD_POINT_MAX else upper - 1

    def _sym_value(self, value: torch.SymInt, point: str) -> int:
        """SymInt を評価点で具体値にする。"""
        if not value.node.expr.free_symbols:
            return int(value)
        expr = self._sym_parts(value)
        return expr.coeff * self._point_value(expr.sym, point) + expr.offset

    def _assign_input_symbols(self) -> None:
        """user 入力 placeholder の出現順で IR シンボル名を割り当てる（唯一の命名経路）。

        命名を消費順（ノード走査）任せにすると、同じグラフでも中間ノードの並びで名前が
        入れ替わりうる。入力 shape は forward の引数順で決まる決定的な列なので、変換前に
        ここで固定する。走査後は封をして、入力に無いシンボルの後付け命名を拒否する。
        """
        for node in self.gm.graph.nodes:
            if node.op != "placeholder" or node.name not in self.user_inputs:
                continue
            val = node.meta.get("val")
            if not isinstance(val, torch.Tensor):
                continue
            for dim in val.shape:
                if isinstance(dim, torch.SymInt) and dim.node.expr.free_symbols:
                    self._sym_parts(dim)
        self.symbols_assigned = True

    # ---- 定数畳み込み ---------------------------------------------------

    def _classify_foldable(self) -> set[str]:
        """定数と shape シンボルだけに依存する部分木のノード名集合（ADR 0010）。

        葉として適格なのは lifted 定数（T 非依存）と SymInt を運ぶノード（sym_size /
        SymInt 演算）だけ。パラメータ/バッファ経由の畳み込みは巨大定数を焼き込みうるので
        不適格にする（重みは initializer のまま運ぶ）。

        MUST: 記号依存を許すのは「Tmax で焼いて prefix スライスで切り出す」表現が入った
        から（ADR 0010）。ただし記号を**値**として消費した部分木は焼いた定数が Tmax でしか
        正しくないので、{@link SYMBOL_EXTENT_ARGS} の extent 位置以外に記号が届いたノードは
        ここで落とす（下流は依存の連鎖で自然に外れる）。残る可換性は
        _check_prefix_commutes の 2 点実測が受け止める。
        """
        foldable: set[str] = set()
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                if node.name in self.lifted_constants:
                    foldable.add(node.name)
                continue
            if node.op != "call_function":
                continue
            val = node.meta.get("val")
            if not isinstance(val, torch.Tensor):
                # テンソルを運ばないノード（sym_size / SymInt 演算）は shape 情報だけに
                # 依存する。記号を含んでも評価点ごとの置換で具体値になるので葉として適格。
                if isinstance(val, torch.SymInt) or (
                    isinstance(val, int) and not isinstance(val, bool)
                ):
                    foldable.add(node.name)
                continue
            if node.target is aten.expand.default:
                # expand は原則 frontier で止める（畳むと実体化が倍数で膨らむ）。ただし
                # **出力 shape が入力と同一の no-op expand** は 1 要素も増やさないので、
                # 畳み込みの連鎖を切る理由が無い（実測: RoPE の `inv_freq[None,:,None]
                # .expand(batch, -1, 1)` が batch=1 でこの形になり、ここで止めると sin / cos が
                # IR 語彙に必要になる — ADR 0016）。
                if not same_extents(node.meta["val"], node.args[0].meta.get("val")):
                    continue
            elif node.target not in FOLDABLE_OPS:
                continue
            if _uses_symbol_as_data(node):
                # 記号がテンソルデータへ昇格した部分木（`scalar_tensor(T)` 等）。焼くと Tmax
                # でだけ正しい定数になり、2 点評価は偶然一致しうる（module docstring の反例）。
                continue
            if all(dep.name in foldable for dep in node.all_input_nodes):
                foldable.add(node.name)
        return foldable

    def _fold_eval(self, node: Node, point: str) -> Any:
        """適格部分木を、指定の評価点（全シンボル一斉置換）で実評価する。"""
        cache = self._fold_cache.setdefault(point, {})
        cached = cache.get(node.name)
        if cached is not None:
            return cached
        if node.op == "placeholder":
            return self.state[self.placeholder_fqn[node.name]]
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            # テンソルを運ばないノードは式の置換で評価する（実行はしない）。
            return self._sym_value(val, point) if isinstance(val, torch.SymInt) else int(val)

        def resolve(arg: Any) -> Any:
            if isinstance(arg, Node):
                return self._fold_eval(arg, point)
            if isinstance(arg, torch.SymInt):
                return self._sym_value(arg, point)
            if isinstance(arg, (list, tuple)):
                return type(arg)(resolve(item) for item in arg)
            return arg

        out = node.target(
            *resolve(tuple(node.args)), **{k: resolve(v) for k, v in node.kwargs.items()}
        )
        cache[node.name] = out
        return out

    def _add_const(self, tensor: torch.Tensor, where: str) -> str:
        """定数テンソルを重複排除して initializer 化し、IR 値名を返す。

        意味論 dtype は f32 と **i32**（ADR 0010 — 相対位置バケット表の添字）。i64 は境界
        正規化で i32 に落とす（値域外は fail loudly）。bool の initializer は語彙に無い。
        """
        data = tensor.detach().contiguous()
        if data.dtype is torch.int64:
            # MUST: 変換点は normalize_boundary_tensor 1 箇所（値域検査つき — ADR 0009）。
            data = normalize_boundary_tensor(data, where)
        storage = _CONST_STORAGE.get(data.dtype)
        if storage is None:
            raise NotImplementedError(
                f"{where}: dtype {tensor.dtype} の定数は IR v1 の initializer にできない"
                f"（{' / '.join(sorted(set(_CONST_STORAGE.values())))} のみ）"
            )
        dtype = DTYPE_NAMES[data.dtype]
        # 逐次 update で食わせる（`tobytes()` の複製 + 連結の複製で定数 1 本ぶんの RAM を
        # 2 度取らない）。食わせるバイト列は「宣言 JSON ‖ 生バイト」のまま**不変**なので
        # digest は 1 ビットも変わらない。
        hasher = hashlib.sha256()
        hasher.update(json.dumps([str(data.dtype), list(data.shape)]).encode())
        hasher.update(memoryview(data.numpy()).cast("B"))
        digest = hasher.hexdigest()
        # MUST: 同一性キーは **full hexdigest**。名前用に短縮した 16 hex（64bit）を突合に
        # 使うと、衝突した 2 つの定数を実体比較なしで 1 本に畳む = 黙って別の値を共用した
        # グラフが出る。名前だけは従来どおり短縮形（IR の可読性）。
        existing = self._const_by_digest.get(digest)
        if existing is not None:
            return existing
        short = digest[:16]
        name = f"const_{short}"
        key = f"const.{short}"
        if name in self.graph.initializers or key in self.tensors:
            # full が違うのに短縮形が一致した（= 名前衝突）。確率的に到達しないが、通せば
            # 上書きで先の定数が消える沈黙誤グラフになるので落とす。
            raise AssertionError(
                f"{where}: 定数名 '{name}' が別の定数と衝突した（digest {digest}）"
            )
        self.tensors[key] = data
        self.graph.initializers[name] = IrInitializer(tensor=key, storage=IrStorage(dtype=storage))
        self.graph.values[name] = IrValue(dtype=dtype, shape=[int(d) for d in data.shape])
        self._const_by_digest[digest] = name
        return name

    def _emit_folded(self, node: Node) -> str:
        """畳み込み frontier のノードを「Tmax 定数 + sym_prefix_slice」として出す（ADR 0010）。

        記号次元が 1 つも無ければ定数そのもの（従来どおり）。あれば、Tmax で焼いた定数の
        prefix を実行時に切り出すノードを 1 本足す。
        """
        where = f"畳み込み定数 {node.name} ({node.target})"
        folded = self._fold_eval(node, FOLD_POINT_MAX)
        val = self._meta_val(node)
        slices: list[dict[str, int]] = []
        symbols: set[str] = set()
        for index, dim in enumerate(val.shape):
            if not isinstance(dim, torch.SymInt) or not dim.node.expr.free_symbols:
                continue
            expr = self._sym_parts(dim)
            symbols.add(expr.sym)
            slices.append({"dim": index, "coeff": expr.coeff, "offset": expr.offset})
        if len(symbols) > 1:
            # sym_prefix_slice の sym はノード単位に 1 つ（IR v1 の attrs 契約）。複数
            # シンボルにまたがる定数は表現を持たないので、黙って畳まず落とす。
            raise NotImplementedError(
                f"{where}: 複数シンボル {sorted(symbols)} にまたがる畳み込み定数は IR v1 非対応"
            )
        sym = next(iter(symbols)) if symbols else None
        self._check_prefix_commutes(node, folded, sym, slices)
        const_name = self._add_const(folded, where)
        # MUST: 焼いた実テンソルの dtype / shape が meta の宣言と食い違っていないか見る。
        # 食い違ったまま通すと、sym_prefix_slice の入出力が別物のバッファを指す。
        declared = self.graph.values[const_name]
        expected_dtype = self._dtype(val, where)
        if declared.dtype != expected_dtype:
            raise AssertionError(
                f"{where}: 焼いた定数の dtype {declared.dtype} が宣言 {expected_dtype} と違う"
            )
        if not slices:
            if declared.shape != self._shape(val):
                raise AssertionError(
                    f"{where}: 焼いた定数の shape {declared.shape} が宣言 {self._shape(val)} と違う"
                )
            return const_name
        # MUST: prefix スライスは strided 実体化コピーで実行する（新カーネルは無い — ADR 0010）。
        # rank 上限は記号次元が 1 本でもある時点で効くので、emit する前にここで落とす
        # （slices が空 = 定数そのものの経路には dispatch が無いので上限も無い）。
        rank = len(val.shape)
        if not 1 <= rank <= STRIDED_RANK:
            raise NotImplementedError(
                f"{where}: rank {rank} の記号 prefix スライスは strided カーネル"
                f"（rank ≤ {STRIDED_RANK}）で実行できない"
            )
        return self._emit(
            "sym_prefix_slice", [const_name], node.name, val, {"sym": sym, "slices": slices}
        )

    def _check_prefix_commutes(
        self, node: Node, folded: Any, sym: str | None, slices: Sequence[Mapping[str, int]]
    ) -> None:
        """畳み込み結果が prefix スライスと可換であることを **2 点評価で実測**する（ADR 0010）。

        シンボルを「値」として使う部分木（`arange(T-1, -1, -1)` や `full([T], T)` 等）は
        allowlist 掲載 op だけで構成できてしまい、Tmax で焼いた定数の先頭要素が実行時の
        正しい値と別物になる — 例外も警告もなく数が変わるので、**宣言ではなく検査で止める**。

        prefix 長は `coeff·sym+offset` で評価する（係数は長さの線形写像なので可換性の意味は
        そのまま）。多シンボルでは第 2 点で全シンボルを同時に動かす（1 つずつ動かすと他を
        Tmax に固定した断面しか見ず、シンボル間の積の形を取り逃す）。

        NOTE: 比較は **ビット厳密**（{@link _bitwise_equal}）。数値等価だと `-0.0 == 0.0` が
        素通りするので、符号ビットまで一致を要求する。NaN は payload まで一致した時だけ通る。
        """
        probed = self._fold_eval(node, FOLD_POINT_PROBE)
        expected = folded
        if sym is not None and isinstance(expected, torch.Tensor):
            bound = self._point_value(sym, FOLD_POINT_PROBE)
            for entry in slices:
                length = entry["coeff"] * bound + entry["offset"]
                expected = expected.narrow(entry["dim"], 0, length)
        if (
            isinstance(probed, torch.Tensor)
            and isinstance(expected, torch.Tensor)
            and _bitwise_equal(probed, expected)
        ):
            return
        points = {
            name: (
                self._point_value(name, FOLD_POINT_MAX),
                self._point_value(name, FOLD_POINT_PROBE),
            )
            for name in self.sym_names.values()
        }
        raise NotImplementedError(
            f"畳み込み部分木が prefix スライスと非可換（シンボルを値として使っている）: "
            f"{node.name} ({node.target}) — Tmax で焼いた定数の prefix と第 2 評価点の"
            f"評価結果が一致しない（シンボル: Tmax, 第 2 点 = {points}）"
        )

    # ---- 変換本体 -------------------------------------------------------

    def _dim(self, dim: Any) -> int | str:
        if isinstance(dim, torch.SymInt):
            if dim.node.expr.free_symbols:
                return format_dim(self._sym_parts(dim))
            return int(dim)
        return int(dim)

    def _shape(self, tensor: torch.Tensor) -> list[int | str]:
        return [self._dim(dim) for dim in tensor.shape]

    def _dtype(self, tensor: torch.Tensor, where: str) -> str:
        name = DTYPE_NAMES.get(tensor.dtype)
        if name is None:
            raise NotImplementedError(
                f"{where}: torch dtype {tensor.dtype} は IR v1 の意味論 dtype 語彙に無い"
                f"（{', '.join(DTYPE_NAMES.values())}）"
            )
        return name

    def _meta_val(self, node: Node) -> torch.Tensor:
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            raise NotImplementedError(f"tensor 以外の値: {node.name} ({node.target})")
        return val

    def _register_value(self, name: str, val: torch.Tensor) -> None:
        self.graph.values[name] = IrValue(
            dtype=self._dtype(val, f"値 '{name}'"), shape=self._shape(val)
        )

    def _emit(
        self,
        op: str,
        ins: list[str],
        out_name: str,
        val: torch.Tensor,
        attrs: Mapping[str, Any],
    ) -> str:
        """CSE しつつノードを追加し、出力値名を返す。"""
        if op not in EMITTABLE_OPS:
            raise AssertionError(f"EMITTABLE_OPS 未登録の IR op: {op}")
        # MUST: CSE キーに attrs を含める。同じ (op, ins) でも attrs が違えば別の値
        # （`cast(x, to=bool)` と `cast(x, to=f32)`）で、落とすと片方が黙って消える。
        # MUST: 出力 shape も含める。reshape / expand は「出力の宣言 shape が目標形」で
        # attrs を持たない（ADR 0011）ため、shape 抜きのキーでは同じ入力の別々の
        # `reshape` が 1 本に畳まれ、片方の消費者が黙って別の形の値を読む。
        shape = self._shape(val)
        key = json.dumps([op, ins, dict(attrs), shape], sort_keys=True)
        cached = self._cse.get(key)
        if cached is not None:
            return cached
        self.graph.nodes.append(IrNode(op=op, ins=list(ins), outs=[out_name], attrs=dict(attrs)))
        self._register_value(out_name, val)
        self._cse[key] = out_name
        return out_name

    def convert(self) -> tuple[IrGraph, dict[str, torch.Tensor]]:
        foldable = self._classify_foldable()
        self._assign_input_symbols()
        self._assert_all_ops_supported(foldable)

        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                self._handle_placeholder(node)
            elif node.op == "call_function":
                if node.name in foldable:
                    continue  # frontier の消費側から評価される（未消費なら自然に DCE）
                self._handle_call(node, foldable)
            elif node.op == "output":
                self.graph.outputs = [
                    self._in_name(out, foldable) for out in self._user_outputs(node)
                ]
            else:
                raise NotImplementedError(f"未対応ノード種: {node.op}")
        # 宣言するのは実際に shape へ出たシンボルだけ（静的グラフは []）。使わないシンボルを
        # 載せるとランタイムが「束縛する入力が無い」で落ちる。
        self.graph.symbols = list(self.sym_names.values())
        # MUST: 変換の出口で全ノードの出力 shape を契約から計算し直して宣言と突き合わせる。
        # 宣言は torch の meta 由来なので、契約の規則（gather の先行次元一致 / conv1d の
        # 出力長 / reshape の要素数など）と食い違う形が「torch 的には合法」として素通りし、
        # ランタイムの Session 構築で初めて落ちる — その差をここで潰す。
        assert_graph_shapes(self.graph)
        return self.graph, self.tensors

    def _assert_all_ops_supported(self, foldable: set[str]) -> None:
        """未対応 aten op を全件集めてから 1 度だけ落とす（部分近似はしない）。"""
        missing: dict[str, list[str]] = {}
        for node in self.gm.graph.nodes:
            if node.op != "call_function" or node.name in foldable:
                continue
            if node.target in SKIP_OPS or node.target in ATEN_HANDLERS:
                continue
            if not isinstance(node.meta.get("val"), torch.Tensor):
                # SymInt/int を運ぶだけの shape 機構（値として使われれば arity 検査が落とす）
                continue
            missing.setdefault(str(node.target), []).append(node.name)
        if missing:
            raise UnsupportedAtenOpsError(missing)

    def _user_outputs(self, out_node: Node) -> list[Node]:
        """output ノードの引数のうち user 出力だけを返す。

        torch.export は USER_OUTPUT の前に BUFFER_MUTATION 等を並べるため、素通しすると
        出力列がずれる。IR v1 は状態更新を表現できないので、黙って捨てず未対応として落とす。
        specs と output 引数の対応は「位置」でのみ有効（normalize は graph_signature を
        更新しないので `output_specs[].arg.name` は実在しないノード名のまま残りうる）。
        """
        specs = self.ep.graph_signature.output_specs
        args = list(out_node.args[0])
        if len(specs) != len(args):
            raise AssertionError(
                f"output_specs と output 引数の本数不一致: {len(specs)} vs {len(args)}"
            )
        other = sorted({s.kind.name for s in specs if s.kind is not OutputKind.USER_OUTPUT})
        if other:
            raise NotImplementedError(f"user 出力以外の出力種別は IR v1 非対応: {other}")
        return [
            arg
            for arg, spec in zip(args, specs, strict=True)
            if spec.kind is OutputKind.USER_OUTPUT
        ]

    def _handle_placeholder(self, node: Node) -> None:
        if node.name in self.user_inputs:
            val = self._meta_val(node)
            # MUST: 入力は inputs[] だけで宣言する（values{} にも書くと IR v1 の
            # 「宣言はちょうど 1 箇所」に反して二重宣言で拒否される — docs/ir-v1.md）。
            self.graph.inputs.append(
                IrInput(
                    name=node.name,
                    dtype=self._dtype(val, f"入力 '{node.name}'"),
                    shape=self._shape(val),
                )
            )
            self.value_of[node.name] = node.name
            return
        if node.name not in self.placeholder_fqn:
            raise NotImplementedError(f"素性不明の placeholder: {node.name}")
        # パラメータ/バッファ/定数 — 実参照された時に _in_name が initializer 化する

    def _materialize_initializer(self, node: Node) -> None:
        fqn = self.placeholder_fqn[node.name]
        tensor = self.state[fqn].detach().contiguous()
        where = f"initializer '{node.name}' ({fqn})"
        if tensor.dtype is not torch.float32:
            raise NotImplementedError(
                f"{where}: dtype {tensor.dtype} は IR v1 の initializer にできない"
                "（意味論 f32 のみ）"
            )
        self.tensors[fqn] = tensor
        self.graph.initializers[node.name] = IrInitializer(
            tensor=fqn, storage=IrStorage(dtype="f32")
        )
        self._register_value(node.name, tensor)
        self.value_of[node.name] = node.name

    def _in_name(self, node: Node, foldable: set[str]) -> str:
        while node.name in self.alias_of:
            node = self.alias_of[node.name]
        existing = self.value_of.get(node.name)
        if existing is not None:
            return existing
        if node.op == "placeholder":
            # user 入力は placeholder 処理で登録済みなので、ここに来るのは initializer のみ
            self._materialize_initializer(node)
            return self.value_of[node.name]
        if node.name in foldable:
            name = self._emit_folded(node)
            self.value_of[node.name] = name
            return name
        raise AssertionError(f"未解決の入力値: {node.name}（トポ順序の破れ）")

    def _tensor_args(self, node: Node) -> list[Node]:
        """位置引数のうちテンソル値の Node をテンソル入力として順に返す。

        all_input_nodes は内部 dict のキーで重複排除されるため使えない — `x*x` のように
        同じ値を両オペランドに取る形は同じ値名が 2 回並ぶのが正しい。SymInt を運ぶだけの
        ノードはテンソルではないので除外し、値として使われた形は入力数検査で落とす。

        MUST: **リスト引数を 1 段だけ平坦化する**（`aten.cat([a, b], dim)` の第 1 引数は
        テンソルの**リスト**で、素通しするとテンソル入力 0 本になる）。ここで拾うのは
        「Node かつ meta がテンソル」だけなので、int のリスト（permute の dims / view の
        sizes）は従来どおり素通りする。
        """
        out: list[Node] = []
        for arg in node.args:
            items = arg if isinstance(arg, (list, tuple)) else (arg,)
            for item in items:
                if isinstance(item, Node) and isinstance(item.meta.get("val"), torch.Tensor):
                    out.append(item)
        return out

    def _check_arity(self, node: Node, spec: Emitted, ins: list[str]) -> None:
        if len(ins) == spec.ins:
            return
        symints = [
            arg.name
            for arg in node.args
            if isinstance(arg, Node) and not isinstance(arg.meta.get("val"), torch.Tensor)
        ]
        hint = f" — SymInt {symints} を値として使う形は IR v1 非対応" if symints else ""
        raise NotImplementedError(
            f"{node.target}: {spec.op} のテンソル入力は {spec.ins} 本のはずが {len(ins)} 本"
            f"{hint} (args={node.args}, kwargs={node.kwargs})"
        )

    def _handle_call(self, node: Node, foldable: set[str]) -> None:
        if node.target in SKIP_OPS:
            check = SKIP_OPS[node.target]
            if check is not None:
                check(node)
            self.alias_of[node.name] = node.all_input_nodes[0]
            return
        if not isinstance(node.meta.get("val"), torch.Tensor):
            # SymInt/int を運ぶだけの shape 機構（sym_size / operator.mul 等）。IR 値は
            # shape 式で表現済み — テンソル位置で消費されれば arity 検査が落とす。
            return
        handler = ATEN_HANDLERS[node.target]  # 未登録は _assert_all_ops_supported が既に落とす
        if any(_holds_node(v) for v in node.kwargs.values()):
            raise NotImplementedError(
                f"{node.target}: テンソルを kwargs で受ける形は未対応 (kwargs={node.kwargs})"
            )
        ins = [self._in_name(arg, foldable) for arg in self._tensor_args(node)]
        spec = handler(node)
        for value, length in spec.synth_consts:
            # MUST: 宣言順に末尾へ足す（省略可能なスロットは常に末尾側 — bias / affine /
            # weight）。合成先は _add_const なので、同じ (値, 長さ) の定数は digest で 1 本に
            # 畳まれる（bias 無し linear が 698 本あっても initializer は長さごとに 1 つ）。
            ins = [
                *ins,
                self._add_const(
                    torch.full((length,), value, dtype=torch.float32),
                    f"{node.target} の定数スロット合成（{node.name}: {value} × {length}）",
                ),
            ]
        self._check_arity(node, spec, ins)
        self.value_of[node.name] = self._emit(
            spec.op, ins, node.name, self._meta_val(node), spec.attrs
        )


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
    dims = node.args[1]
    _expect(dims is not None, node, "全次元 sum は未対応（reduce は 1 軸のみ）")
    keepdim = bool(node.args[2]) if len(node.args) > 2 else bool(node.kwargs.get("keepdim", False))
    dtype = node.args[3] if len(node.args) > 3 else node.kwargs.get("dtype")
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
        dims = list(node.args[1]) if len(node.args) > 1 else []
        keepdim = (
            bool(node.args[2]) if len(node.args) > 2 else bool(node.kwargs.get("keepdim", False))
        )
        # 軸は 1 本だけ（sum と同じ絞り方 — 実行側は 1 軸ずつの縮約しか持たない）。
        _expect(len(dims) == 1, node, f"複数軸 {dims} の {op} は未対応（1 軸ずつ）")
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
    enable_gqa=False)`。**受理するのは「非因果・dropout 0・GQA 無し・rank-4」で、マスクは
    無しか f32 加算型 `[1,1,M,N]` だけ**。残りは全件列挙して fail loudly にする
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
    enable_gqa = _arg_or_kwarg(node, 7, "enable_gqa", False)
    _expect(
        enable_gqa is False,
        node,
        "enable_gqa=True の attention は未対応（KV head の複製は契約に無い）",
    )
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
    # perf-a（ADR 0023）— SDPA を保存したターゲットだけがこのハンドラに来る
    # （`curated_decompositions(preserved=…)` がターゲット別に選ぶ）。
    aten.scaled_dot_product_attention.default: _h_attention,
}


def convert(
    ep: ExportedProgram, *, symbol_names: Sequence[str] = ("T",)
) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """IR グラフと格納テンソル辞書を返す。

    symbol_names は IR に載るシンボル名を user 入力 placeholder の出現順
    （= forward の引数順）で与える。
    """
    return Converter(ep, symbol_names=symbol_names).convert()


def curated_decompositions(preserved: Sequence[str] = PRESERVED_OP_PREFIXES):
    """分解表から「保存する高位 op」を外して返す。

    core 分解するとカーネル粒度と 1 対 1 でなくなる op を default_decompositions から
    取り除く。既定は 11 op（PRESERVED_OP_PREFIXES — ADR 0007 の 9 op に ADR 0015 で
    leaky_relu、ADR 0017 で rms_norm を追加）。

    NOTE: `rms_norm` は保存だけでは足りない。`aten.rms_norm` を出すのは diffusers の
    `nn.RMSNorm` 経由だけで、手書き分解形（Qwen3 / DiT）は normalize の `_fold_rms_norm` が
    合成する（ADR 0016 / 0017 の供給 2 系統）。

    NOTE: SDPA（ADR 0023）は既定に**含めない** — 12 本目は
    {@link PRESERVED_OP_PREFIXES_WITH_ATTENTION} を明示的に渡したターゲットだけが得る。
    """
    from torch.export import default_decompositions

    table = default_decompositions()
    for overload in list(table):
        if str(overload).startswith(tuple(preserved)):
            del table[overload]
    return table
