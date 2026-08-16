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
   対応表そのもの（ハンドラ群）は {@link karume.aten_handlers} — ここはエンジン側。
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
from typing import Any

import torch
from torch.export import ExportedProgram
from torch.export.graph_signature import OutputKind
from torch.fx import Node

# 対応表側（{@link karume.aten_handlers}）へ出した共有物の束縛。import は convert →
# aten_handlers の一方向のみ（逆辺を張ると対応表がエンジンへ食い込む）。
from karume.aten_handlers import (
    ATEN_HANDLERS,
    DTYPE_NAMES,
    Emitted,
    _expect,
    _has_free_symbols,
    aten,
)
from karume.dims import MAX_SAFE_INT, DimExpr, format_dim, is_symbol_name
from karume.extents import same_extents
from karume.ir import (
    IrGraph,
    IrInitializer,
    IrInput,
    IrNode,
    IrStorage,
    IrValue,
)
from karume.ops import EMITTABLE_OPS, STRIDED_RANK
from karume.shapes import assert_graph_shapes

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
