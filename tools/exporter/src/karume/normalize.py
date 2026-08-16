"""curated decomp 後の FX グラフ正規化 — IR 語彙を増やさない同値書き換え。

convert() の前段で呼ぶ。各パスは `(graph, stats) -> None` の関数で、`_passes()` が並べた
順に走る（順序に依存があるパスを足すときはここにコメントで理由を残すこと）。

- `_drop_metadata_asserts`: `aten._assert_tensor_metadata` の削除。eliminate_dead_code は
  これを impure と見なして残すため、置き換え済みの部分木が assert 経由で生き残る。
- `_fold_rms_norm`: `mul(weight, mul(x, rsqrt(mean(x²)+eps)))` → `rms_norm` 1 ノード
  （ADR 0016 / 0017。手書き RMSNorm は preserve では畳めない — FX パターンマッチが要る）。
- `_drop_safe_softmax_guard`: SDPA の safe-softmax ガード（`eq(-inf)` / `any.dim` /
  `full_like` / `where`）を畳む。不活性を証明できたら**除去**（ADR 0016）、できなければ
  `safe_softmax` へ**構成的に置換**する（ADR 0044）。
- `_additive_attn_mask`: 保存した SDPA の bool `attn_mask` を加算型 f32（0 / −inf）へ
  （ADR 0023 の mask 契約。torch 自身の bool → additive 変換と同じ op・同じ定数）。
- `_lower_unit_expand`: GQA repeat_kv の `unsqueeze → expand → (clone) → view` を rank≤3 へ。
- `_lower_split_unbind`: RoPE の「最終次元 split → 幅 1 slice → squeeze」を最終次元 slice へ
  （消費者が揃わない形は書き換えずスキップ）。
- `_lower_reshape_permute`: rank≥5 の `reshape → permute → reshape`（patchify / unpatchify）を
  rank4 の隣接転置列へ分解する。
- `_compose_permute_chains`: 恒等 `permute` の除去と、隣接する `permute` の合成（p∘q）。
  中間の実体化コピーと dispatch がそのぶん消える。
- `_pow2_to_mul`: `pow.Tensor_Scalar(x, 2)` → `mul(x, x)`。M0 語彙に pow は無いが、
  同値な mul はある（数値も同一 — 乗算 1 回に落ちるだけ）。
- `_drop_identity_repeat`: `repeat(x, [1,…,1])` → x（recon §4-1。実測 4 本が全てこの形）。
- `_drop_identity_add`: `add(x, 0)` → x（recon §4-2。c2p / p2c 加算の初期値 2 本）。
- `_promote_scalar_operands`: `add/sub/mul/div(tensor, スカラ)` → 二項 op + rank-1 定数
  （recon §4-4。IR 語彙も attrs も増やさない唯一の道）。
- `_eq_zero_to_not_bool`: `eq(x, 0)` → `bitwise_not(cast(x, bool))`（ADR 0015 の裁定。
  cast 規約「x → bool は x != 0」からの帰結で、新 op を作らずに済む）。
- `_select_to_squeeze`: `select.int(長さ 1 の軸, 添字 0)` → `squeeze`（汎用衛生）。
- `_split_to_slices`: `split_with_sizes` + `getitem` → `slice` の列（ADR 0014。IR に多出力 op が
  無いので、分割は取り出し口ごとの slice に開く）。

MUST: `_fold_rms_norm` は `_pow2_to_mul` と `_promote_scalar_operands` の**両方より先**に
走る（ADR 0016 の順序 MUST）。`pow(x,2)` が `mul(x,x)` に潰れてもパターン照合自体は
`_is_square_of` が両形を見るので生き残るが、`add(mean, eps)` の eps が rank-1 定数へ昇格すると
Python スカラの照合が外れて畳めなくなる。

MUST: 鎖 3 パス（rank 下げ）の発火は **rank > STRIDED_RANK の値を含む形に限る**（ADR 0016）。
既存グラフ（SBV2 / DeBERTa は全て rank ≤ 4）へ誤爆させないための安全線で、新しい定数は
作らない — 「strided カーネルが実行できない rank」が発火条件そのものだから。

MUST: `_drop_identity_add` は `_promote_scalar_operands` より**先**に走る。逆順にすると
`add(x, 0)` が `add(x, const 0)` になり、恒等除去の対象（Python スカラの 0）でなくなって
無駄な定数と 1 dispatch が残る。

MUST: ExportedProgram の graph_signature は更新しない。出力ノードを書き換えると
`output_specs[].arg.name` は実在しないノード名のまま残るが、唯一の消費者である
convert._user_outputs が specs と output 引数を「位置」で対応づけるため無害
（名前を同期して持つのは派生状態の二重管理になる）。
"""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch.export import ExportedProgram
from torch.fx import Graph, Node

from karume.extents import extent_key, extent_keys
from karume.ops import STRIDED_RANK

aten = torch.ops.aten

#: safe-softmax ガードの不活性を実測する 2 評価点（記号長のマスク用）。2 点で見るのは
#: 「たまたま片方の長さで成立した」を弾くため（convert._check_prefix_commutes と同じ流儀）。
GUARD_PROBE_LENGTHS = (5, 9)

NEG_INF = float("-inf")

#: safe-softmax ガードを構成的に置換した softmax ノードに立てる旗（ADR 0044）。
#:
#: FX グラフ層に "safe_softmax" という aten op は無いので、ガードを畳んだ事実は元の softmax
#: ノードの meta で運び、`aten_handlers._h_softmax` が op 名の分岐に使う。**meta を消すパスを
#: このパスより後ろに置かないこと** — 旗が落ちると素の softmax として焼かれ、全 -inf 行が
#: 黙って NaN になる。
SAFE_SOFTMAX_META = "karume_safe_softmax"

#: 論理的な要素順を変えない reshape 系（rank 下げパスが自由に括り直してよい形）。
_RESHAPE_OPS = (
    aten.view.default,
    aten.squeeze.dim,
    aten.squeeze.dims,
    aten.unsqueeze.default,
)


def _val(node: Node) -> torch.Tensor:
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        raise NotImplementedError(f"tensor 以外の値を rank 下げ対象にできない: {node.name}")
    return val


def _rank(node: Any) -> int:
    val = node.meta.get("val") if isinstance(node, Node) else None
    return val.dim() if isinstance(val, torch.Tensor) else 0


def _static_size(dim: Any) -> int | None:
    """次元長の具体値（記号を含むなら None）。

    MUST: SymInt を `int()` で採らない — export 時のヒント値へ黙って特殊化され、
    「ヒント値 1 の記号次元」が squeeze に化けて実行時に無言の誤値になる。
    """
    if isinstance(dim, torch.SymInt):
        return None if dim.node.expr.free_symbols else int(dim)
    return int(dim)


def _numel(shape: Any) -> Any:
    return math.prod([1, *shape])


def _is_reshape(node: Any) -> bool:
    return isinstance(node, Node) and node.op == "call_function" and node.target in _RESHAPE_OPS


def _sole_user(node: Node) -> Node | None:
    users = list(node.users)
    return users[0] if len(users) == 1 else None


def _insert(graph: Graph, anchor: Node, target: Any, args: tuple, ref: torch.Tensor, **kwargs: Any):
    """`anchor` の直前にノードを挿し、meta["val"] を fake mode で合成する。

    convert は全ノードの `meta["val"]`（FakeTensor）を要求するので、挿入ノードにも
    宣言を持たせないと変換の出口（assert_graph_shapes）まで届かない。
    """
    with graph.inserting_before(anchor):
        node = graph.call_function(target, args, kwargs)
    resolved = tuple(a.meta["val"] if isinstance(a, Node) else a for a in args)
    with ref.fake_mode:
        node.meta["val"] = target(*resolved, **kwargs)
    return node


def _insert_contiguous(graph: Graph, anchor: Node, src: Node, ref: torch.Tensor) -> Node:
    """permute / expand の直後に置く実体化コピー。

    IR 側では permute / expand が strided 実体化コピーなので意味論上は冗長（convert の
    SKIP_OPS が恒等として吸収する）だが、これを挟まないと後続 view の meta 合成
    （torch の view 規則）が通らない。
    """
    return _insert(
        graph, anchor, aten.clone.default, (src,), ref, memory_format=torch.contiguous_format
    )


def _as_contiguous(graph: Graph, anchor: Node, src: Node, ref: torch.Tensor) -> Node:
    """view を挟む前に非連続な値を実体化する（連続ならそのまま）。"""
    return src if _val(src).is_contiguous() else _insert_contiguous(graph, anchor, src, ref)


def _replace(graph: Graph, old: Node, new: Node) -> None:
    old.replace_all_uses_with(new)
    graph.erase_node(old)


def _drop_metadata_asserts(graph: Graph, stats: Counter) -> None:
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target is aten._assert_tensor_metadata.default:
            graph.erase_node(node)
            stats["drop_metadata_assert"] += 1


# ---- RMS 正規化パターンの畳み込み（ADR 0016 / 0017）------------------------


def _eps_operand(node: Any) -> float | None:
    """`add(mean, eps)` の eps を返す（`add.Tensor` のスカラ形と `add.Scalar` の両方）。"""
    if not isinstance(node, Node) or node.op != "call_function":
        return None
    if node.target not in (aten.add.Tensor, aten.add.Scalar):
        return None
    if node.kwargs.get("alpha", 1) != 1 or len(node.args) != 2:
        return None
    rhs = node.args[1]
    if isinstance(rhs, bool) or not isinstance(rhs, (int, float)):
        return None
    return float(rhs)


def _is_last_dim_mean(node: Any) -> bool:
    """最終次元 1 本・keepdim の `mean.dim` か（rms_norm の縮約軸は最終次元のみ）。"""
    if not isinstance(node, Node) or node.op != "call_function" or node.target is not aten.mean.dim:
        return False
    src = _val(node.args[0])
    dims = [int(d) for d in node.args[1]]
    keepdim = bool(node.args[2]) if len(node.args) > 2 else bool(node.kwargs.get("keepdim", False))
    return keepdim and dims in ([-1], [src.dim() - 1])


def _is_square_of(node: Any, x: Node) -> bool:
    """x² を表すノードか。decomp の当たり方で `pow(x,2)` と `mul(x,x)` の両形が出る。"""
    if not isinstance(node, Node) or node.op != "call_function":
        return False
    if node.target is aten.pow.Tensor_Scalar:
        return node.args[0] is x and node.args[1] == 2
    return node.target is aten.mul.Tensor and node.args[0] is x and node.args[1] is x


def _rms_norm_operands(node: Node) -> tuple[Node, Node, float] | None:
    """`mul(weight, mul(x, rsqrt(mean(x²)+eps)))` を (x, weight, eps) に分解する。

    2 つの mul はどちらも可換なので、両方の並びを試す（実測は Qwen3 が
    `mul(mul(x, rsqrt), weight)`、DiT の QK ノルムが逆順）。
    """
    if node.op != "call_function" or node.target is not aten.mul.Tensor:
        return None
    for scaled, weight in (node.args[:2], node.args[1::-1]):
        if not isinstance(scaled, Node) or not isinstance(weight, Node):
            continue
        if scaled.op != "call_function" or scaled.target is not aten.mul.Tensor:
            continue
        for x, rsqrt in (scaled.args[:2], scaled.args[1::-1]):
            if not isinstance(x, Node) or not isinstance(rsqrt, Node):
                continue
            if rsqrt.op != "call_function" or rsqrt.target is not aten.rsqrt.default:
                continue
            eps = _eps_operand(rsqrt.args[0])
            if eps is None:
                continue
            mean = rsqrt.args[0].args[0]
            if not _is_last_dim_mean(mean) or not _is_square_of(mean.args[0], x):
                continue
            return x, weight, eps
    return None


def _fold_rms_norm(graph: Graph, stats: Counter) -> None:
    """RMS 正規化の分解形を `aten.rms_norm` 1 ノードへ畳む（ADR 0017 の供給ルート②）。

    手書き実装（Qwen3 / DiT）と diffusers `nn.RMSNorm` は core decomp で同じ
    pow/mean/rsqrt 形に降りるので、供給元を問わず 1 パターンで拾える。

    MUST: weight が**最終次元長の rank-1** でない形は畳まない。rms_norm の契約は
    「正規化長の正本 = weight の長さ」（ADR 0017）で、broadcast 形の weight を畳むと
    契約と別の意味論になる。畳まなければ mean/rsqrt が未対応 op として全件列挙に出る
    （黙って近似しない）。
    MUST: eps は有限の正数。非有限 eps は IR v1 が値レベルで拒否する。
    """
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        matched = _rms_norm_operands(node)
        if matched is None:
            continue
        x, weight, eps = matched
        x_val, weight_val = _val(x), _val(weight)
        length = _static_size(x_val.shape[-1])
        if length is None or list(weight_val.shape) != [length]:
            continue
        if not math.isfinite(eps) or eps <= 0:
            continue
        new = _insert(graph, node, aten.rms_norm.default, (x, [length], weight, eps), x_val)
        # 出力は元の mul と同形・同 dtype（meta を正本として持ち越す）。
        new.meta.update(node.meta)
        _replace(graph, node, new)
        stats["rms_norm"] += 1


# ---- safe-softmax ガードの除去（ADR 0016）----------------------------------


def _introduces_neg_inf(node: Node) -> bool:
    def holds(value: Any) -> bool:
        if isinstance(value, float):
            return value == NEG_INF
        if isinstance(value, (list, tuple)):
            return any(holds(item) for item in value)
        return False

    return holds(node.args) or holds(tuple(node.kwargs.values()))


#: 入力が -inf でも出力が有限（か NaN）になる op — -inf の伝播はここで止まる。
#: softmax は「全要素 -inf の行」を NaN にするが -inf にはしない。NaN は `eq(-inf)` で
#: 拾われないのでガードの発火条件には効かない（全行有限の証明は各段で個別に立てる）。
_NEG_INF_SANITIZERS = (
    aten.softmax.int,
    aten._softmax.default,
    aten.sigmoid.default,
    aten.tanh.default,
    aten.exp.default,
)


class _Placeholders:
    """placeholder の実値表（safe-softmax ガードの不活性証明が使う — ADR 0016）。

    torch.export は parameter / buffer / lifted 定数を**全て placeholder として持ち上げる**
    ので、「placeholder は有限」と決め打つと `register_buffer` した -inf マスクが素通りする。
    実値は ExportedProgram（`state_dict` / `constants`）から引ける。

    唯一の例外がユーザー実行時入力（`graph_signature.user_inputs`）で、こちらは値が実行時に
    しか決まらないため**有限と仮定する**。グラフ入力に -inf を渡す形はランタイム契約の外で、
    IR v1 の非有限値拒否は initializer にしか効かない。

    実値を引けない placeholder は「-inf 源かもしれない」に倒す（呼び出し側が fail loudly）。
    有限判定は**遅延**する — 判定に触れない巨大な重みまで走査すると export が数秒遅くなる。
    """

    def __init__(self, ep: ExportedProgram) -> None:
        signature = ep.graph_signature
        targets = {
            **signature.inputs_to_parameters,
            **signature.inputs_to_buffers,
            **signature.inputs_to_lifted_tensor_constants,
        }
        user_inputs = set(signature.user_inputs)
        self._values: dict[str, torch.Tensor] = {}
        self._unknown: set[str] = set()
        for node in ep.graph_module.graph.nodes:
            if node.op != "placeholder" or node.name in user_inputs:
                continue
            target = targets.get(node.name)
            value = ep.state_dict.get(target, ep.constants.get(target)) if target else None
            if isinstance(value, torch.Tensor):
                self._values[node.name] = value
            else:
                self._unknown.add(node.name)
        self._finite: dict[str, bool] = {}

    def value(self, name: str) -> torch.Tensor | None:
        """placeholder の実値（user 入力・実値不明なら None）。"""
        return self._values.get(name)

    def can_be_neg_inf(self, name: str) -> bool:
        """この placeholder が非有限値を持ちうるか（user 入力は有限仮定で False）。"""
        if name in self._unknown:
            return True
        value = self._values.get(name)
        if value is None:
            return False
        cached = self._finite.get(name)
        if cached is None:
            cached = not value.is_floating_point() or bool(torch.isfinite(value).all())
            self._finite[name] = cached
        return not cached


def _can_be_neg_inf(node: Node, placeholders: _Placeholders) -> bool:
    """node の値が -inf を取り得るかを依存を遡って判定する。

    多層 attention では下段の softmax 経由でマスクが上段の依存錐に入るため、「依存錐に
    -inf 定数があるか」だけでは全段が偽陽性になる。値域が有限な op を伝播の壁として扱い、
    実際に -inf になり得る経路だけを見る。持ち上げられた定数 / buffer / parameter は
    {@link _Placeholders} が実値で判定し、それ以外の非 call_function ノードは
    「-inf 源かもしれない」に倒す。
    """
    seen: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current.name in seen:
            continue
        seen.add(current.name)
        if current.op == "placeholder":
            if placeholders.can_be_neg_inf(current.name):
                return True
            continue
        if current.op != "call_function":
            return True
        val = current.meta.get("val")
        if isinstance(val, torch.Tensor) and val.dtype is torch.bool:
            continue
        if current.target in _NEG_INF_SANITIZERS:
            continue
        if _introduces_neg_inf(current):
            return True
        stack.extend(current.all_input_nodes)
    return False


def _subs_symint(value: Any, length: int) -> Any:
    if not isinstance(value, torch.SymInt):
        return value
    expr = value.node.expr
    if not expr.free_symbols:
        return int(expr)
    symbols = list(expr.free_symbols)
    if len(symbols) != 1:
        raise NotImplementedError(f"複数シンボルの式は評価できない: {expr}")
    return int(expr.subs(symbols[0], length))


def _eval_static(
    node: Node, length: int, cache: dict[str, Any], placeholders: _Placeholders
) -> Any:
    """定数と shape シンボルだけに依存する部分木を、記号長 `length` で実評価する。

    持ち上げられた定数 / buffer / parameter は実値に解決する（`register_buffer` された
    マスクは「入力値に依存する」形ではなく、export 時に値が確定している定数だから）。

    MUST: ユーザー入力の placeholder に触れた時点で失敗する。実行時入力に依存するマスクは
    「不活性の証明」ができない形なので、黙って通さず呼び出し側が fail loudly する。
    """
    cached = cache.get(node.name)
    if cached is not None:
        return cached
    if node.op == "placeholder":
        value = placeholders.value(node.name)
        if value is None:
            raise NotImplementedError(f"実値を持たない placeholder: {node.name}")
        return value
    if node.op != "call_function":
        raise NotImplementedError(f"定数部分木でないノード: {node.name} ({node.op})")
    val = node.meta.get("val")
    if isinstance(val, torch.SymInt):
        return _subs_symint(val, length)

    def resolve(arg: Any) -> Any:
        if isinstance(arg, Node):
            return _eval_static(arg, length, cache, placeholders)
        if isinstance(arg, torch.SymInt):
            return _subs_symint(arg, length)
        if isinstance(arg, (list, tuple)):
            return type(arg)(resolve(item) for item in arg)
        return arg

    out = node.target(*resolve(tuple(node.args)), **{k: resolve(v) for k, v in node.kwargs.items()})
    cache[node.name] = out
    return out


def _guard_parts(node: Node) -> tuple[Node, Node] | None:
    """`where(logical_not(any(logical_not(eq(src,-inf)), -1)), full_like(sm,0), sm)` を照合する。

    一致したら (softmax ノード, softmax の入力) を返す。SDPA の safe-softmax ガード
    （「全要素が -inf の行は 0 を返す」）そのもので、厳密照合以外は触らない。
    """
    if node.op != "call_function" or node.target is not aten.where.self:
        return None
    if len(node.args) != 3:
        return None
    cond, zeros, softmax = node.args
    if not all(isinstance(arg, Node) for arg in (cond, zeros, softmax)):
        return None
    if softmax.target is not aten.softmax.int or zeros.target is not aten.full_like.default:
        return None
    if zeros.args[0] is not softmax or zeros.args[1] != 0:
        return None
    if cond.target is not aten.logical_not.default:
        return None
    reduced = cond.args[0]
    if not isinstance(reduced, Node) or reduced.target is not aten.any.dim:
        return None
    if reduced.args[1] not in (-1, _rank(reduced) - 1):
        return None
    if not (len(reduced.args) > 2 and reduced.args[2]):
        return None
    negated = reduced.args[0]
    if not isinstance(negated, Node) or negated.target is not aten.logical_not.default:
        return None
    equality = negated.args[0]
    if not isinstance(equality, Node) or equality.target is not aten.eq.Scalar:
        return None
    if equality.args[1] != NEG_INF:
        return None
    src = softmax.args[0]
    return (softmax, src) if equality.args[0] is src else None


class _UnprovenGuardError(NotImplementedError):
    """不活性の判定そのものが実評価の例外で立たなかった形（`safe_softmax` へ回す）。

    `NotImplementedError` の派生にしてあるので呼び出し側の受け口は 1 つのまま。統計だけ
    分けるのは、**例外を黙って吸収する経路を作らない**ため（本物の実装バグがここへ紛れ込んだ
    ときに件数として見える）。
    """


def _assert_guard_inactive(src: Node, placeholders: _Placeholders) -> str:
    """ガードが発火し得ない（全要素 -inf の行が存在しない）ことを示し、根拠名を返す。

    ① softmax 入力の依存錐に -inf を持ち込むノードが無い → 有限値しか来ないので発火し得ない
    ② -inf が「加算マスク 1 本」からのみ入る → そのマスクを 2 つの記号長で実評価し、各行に
       有限要素が残ることを実測する（因果マスクの対角がこれで通る）

    MUST: それ以外は `NotImplementedError`。ガードを消すと NaN が下流に流れる形をここで
    受理しない（呼び出し側が `safe_softmax` への構成的置換へ回す — ADR 0044。この関数が
    返すのは**あくまで「消してよい」証明**で、判定を緩めてはならない）。
    """
    if not _can_be_neg_inf(src, placeholders):
        return "no-neg-inf"
    if src.op != "call_function" or src.target is not aten.add.Tensor:
        raise NotImplementedError(
            f"safe-softmax ガードの不活性を証明できない: {src.name} ({src.target})"
            " — -inf が加算マスク以外の経路から入っている"
        )
    left, right = src.args[0], src.args[1]
    if not isinstance(left, Node) or not isinstance(right, Node):
        raise NotImplementedError(f"safe-softmax ガードの加算がテンソル同士でない: {src.name}")
    scores, mask = (left, right) if _can_be_neg_inf(right, placeholders) else (right, left)
    if _can_be_neg_inf(scores, placeholders):
        raise NotImplementedError(
            f"safe-softmax ガード: スコア側にも -inf 源がある（{scores.name}）"
        )
    for length in GUARD_PROBE_LENGTHS:
        try:
            evaluated = _eval_static(mask, length, {}, placeholders)
            rows_have_finite = bool(torch.isfinite(evaluated).any(dim=-1).all())
        except NotImplementedError:
            # 判定関数が構造で下した「証明できない」— そのまま呼び出し側へ返す。
            raise
        except Exception as cause:
            # 実評価は placeholder を**実値**へ解決する一方、記号は probe 長へ置換するので、
            # 両者が同じ式で出会うと torch が例外を投げる（shape 不一致・非テンソル）。
            # これは判定の失敗であってエクスポータの故障ではない — 例外の型名簿を呼び出し側へ
            # 漏らさないよう、判定結果（`NotImplementedError`）へ正規化する MUST。
            raise _UnprovenGuardError(
                f"safe-softmax ガードの不活性を判定できない: マスク {mask.name} の記号長"
                f" {length} での実評価が {type(cause).__name__} で落ちた"
            ) from cause
        if not rows_have_finite:
            raise NotImplementedError(
                f"safe-softmax ガードは不活性でない: マスク {mask.name} は記号長 {length} で"
                " 全要素 -inf の行を持つ"
            )
    return "masked-rows-nonempty"


def _drop_safe_softmax_guard(graph: Graph, stats: Counter, *, placeholders: _Placeholders) -> None:
    """SDPA の safe-softmax ガードを畳む（ADR 0016 → ADR 0044 で 2 段化）。

    ① 不活性を実値で証明できたらガードを**取り除く**（ADR 0016 の従来経路 — 既存資産の
       出力バイト列はここで閉じたまま）。
    ② 証明できない形（実行時マスク・スコア側の -inf 源・全 -inf 行が実在する静的マスク）は
       fail loudly せず、ガード部分木ごと **safe_softmax 1 ノードへ書き換える**。
       グラフ操作は ① と同じ「where → softmax」の置換で、違いは softmax に
       {@link SAFE_SOFTMAX_META} を立てることだけ。

    ② が正しいのは、スコアが有限（非有限入力は契約外）の下で「全要素 -inf ⇔ 行 max = -inf」
    であり、safe_softmax の契約（行 max が -inf の行は全 0）が torch のガードと**厳密に**
    同じ意味論だから（ADR 0044 決定 2）。① を先に試すのは、証明が立つ形で語彙を増やさない
    ため（既存 IR との差分ゼロ）。
    """
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        parts = _guard_parts(node)
        if parts is None:
            continue
        softmax, src = parts
        try:
            reason = _assert_guard_inactive(src, placeholders)
        except _UnprovenGuardError:
            # 判定が実評価の例外で立たなかった形（構造で「証明できない」と言われた形と
            # 区別して数える）。置換そのものは同じ。
            softmax.meta[SAFE_SOFTMAX_META] = True
            reason = "rewritten-unproven"
        except NotImplementedError:
            softmax.meta[SAFE_SOFTMAX_META] = True
            reason = "rewritten-safe"
        stats[f"softmax_guard:{reason}"] += 1
        _replace(graph, node, softmax)


def _additive_attn_mask(graph: Graph, stats: Counter) -> None:
    """保存した SDPA の **bool** な `attn_mask` を加算型 f32（0 / −inf）へ置き換える。

    融合 attention の mask 契約は加算型 f32 のみ（ADR 0023）。一方 transformers の
    Gemma3 系は bool の帯マスクを SDPA に渡すので、保存すると ① `_h_attention` の dtype 門
    ② bool 定数を initializer にできない門、の 2 つに当たる。

    書き換えは `where(mask, scalar_tensor(0.0), scalar_tensor(-inf))` の 1 段で、これは
    torch 自身の `_scaled_dot_product_attention_math` が bool マスクに対して**同じ op・
    同じ定数**で行う変換そのもの（分解経路が実グラフに出す形と一致 — ADR 0023 の
    ビット同一が保存経路でもそのまま成り立つ）。where / scalar_tensor はどちらも
    `convert.FOLDABLE_OPS` にあるので、定数畳み込みで f32 の Tmax 定数 +
    `sym_prefix_slice` に落ちる。

    MUST: 同じ bool マスクには**同じ 1 本**の加算マスクを配る。SDPA ごとに複製しても値は
    同じ（CSE と定数の重複排除で最後は 1 本に畳まれる）が、畳み込み評価を無駄に層数ぶん
    走らせることになる。
    """
    additive: dict[str, Node] = {}
    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or node.target is not aten.scaled_dot_product_attention.default
        ):
            continue
        mask = node.args[3] if len(node.args) > 3 else node.kwargs.get("attn_mask")
        if not isinstance(mask, Node) or _val(mask).dtype is not torch.bool:
            continue
        replacement = additive.get(mask.name)
        if replacement is None:
            ref = _val(mask)
            keep = _insert(
                graph, node, aten.scalar_tensor.default, (0.0,), ref, dtype=torch.float32
            )
            drop = _insert(
                graph, node, aten.scalar_tensor.default, (NEG_INF,), ref, dtype=torch.float32
            )
            replacement = _insert(graph, node, aten.where.self, (mask, keep, drop), ref)
            additive[mask.name] = replacement
        node.replace_input_with(mask, replacement)
        stats["additive_attn_mask"] += 1


# ---- rank 下げ 3 パス（発火は rank > STRIDED_RANK 限定 — ADR 0016）----------


def _lower_unit_expand(graph: Graph, stats: Counter) -> None:
    """`unsqueeze → expand → (clone) → view` を rank≤3 の expand 列に落とす。

    GQA の kv 複製（`repeat_interleave(n, dim)` の分解形）が rank5 を作る唯一の形。
    新しい軸の前後をまとめて `[P, n, Q]` に潰せば複製の意味論はそのまま rank3 になる。
    `n == 1`（GQA 無し構成）は鎖まるごと恒等。
    """
    for expand in list(graph.nodes):
        if expand.op != "call_function" or expand.target is not aten.expand.default:
            continue
        if _rank(expand) <= STRIDED_RANK:
            continue
        unsqueeze = expand.args[0]
        if not isinstance(unsqueeze, Node) or unsqueeze.target is not aten.unsqueeze.default:
            continue
        sink = _sole_user(expand)
        if sink is not None and sink.target is aten.clone.default:
            sink = _sole_user(sink)
        if sink is None or sink.target is not aten.view.default:
            continue
        src = unsqueeze.args[0]
        axis = int(unsqueeze.args[1]) % _rank(unsqueeze)
        src_shape = list(_val(src).shape)
        expanded_shape = list(_val(expand).shape)
        if expanded_shape[:axis] != src_shape[:axis]:
            continue
        if expanded_shape[axis + 1 :] != src_shape[axis:]:
            continue
        count = _static_size(expanded_shape[axis])
        if count is None:
            raise NotImplementedError(f"記号次元の複製は未対応: {expand.name}")
        out_shape = list(_val(sink).shape)
        if count == 1 and extent_keys(out_shape) == extent_keys(src_shape):
            _replace(graph, sink, src)
            stats["unit_expand->identity"] += 1
            continue
        head, tail = _numel(src_shape[:axis]), _numel(src_shape[axis:])
        ref = _val(src)
        flat = _insert(
            graph,
            sink,
            aten.view.default,
            (_as_contiguous(graph, sink, src, ref), [head, 1, tail]),
            ref,
        )
        wide = _insert(graph, sink, aten.expand.default, (flat, [head, count, tail]), ref)
        wide = _insert_contiguous(graph, sink, wide, ref)
        merged = _insert(graph, sink, aten.view.default, (wide, out_shape), ref)
        _replace(graph, sink, merged)
        stats["unit_expand->rank3"] += 1


def _unbind_consumers(view: Node, *, split_dim: int) -> list[tuple[Node, int, int]] | None:
    """`view` の消費者が全て「分割軸の幅 1 slice → squeeze」なら `(squeeze, start, end)` の列。

    1 件でも当てはまらなければ None（= この view は unbind ではない）。
    """
    rank = _rank(view)
    rewritten: list[tuple[Node, int, int]] = []
    for user in list(view.users):
        if user.target is not aten.slice.Tensor or len(user.args) < 4:
            return None
        if int(user.args[1]) % rank != split_dim:
            return None
        start, end = int(user.args[2]), int(user.args[3])
        squeeze = _sole_user(user)
        if end - start != 1 or squeeze is None:
            return None
        if squeeze.target not in (aten.squeeze.dim, aten.squeeze.dims):
            return None
        rewritten.append((squeeze, start, end))
    return rewritten


def _lower_split_unbind(graph: Graph, stats: Counter) -> None:
    """`view`（最終次元を (A,B) に分割）→ 幅 1 `slice` → `squeeze` を最終次元 slice に畳む。

    `apply_rotary_emb(use_real_unbind_dim=-2)` が作る形。分割は連続なので、添字 s の
    1 枚は元テンソルの最終次元 `[s·B, (s+1)·B)` と厳密に同じ並びになる。

    MUST: 消費者の照合は**全件そろってから**書き換える（分割の一部だけを書き換えると残りが
    壊れた参照になる）。1 件でも想定外（幅 1 でない slice / squeeze 以外の消費者）なら
    **その view には触れずスキップ**する — 掴んだ形は unbind とは限らず、ここで落とすと
    別の形の rank5 が正規化の途中で殺される。本当に rank ≤ STRIDED_RANK へ落ちない形が
    残れば convert 側の門（strided 族の rank 上限）が全件列挙で落とす。
    """
    for view in list(graph.nodes):
        if view.op != "call_function" or view.target is not aten.view.default:
            continue
        if _rank(view) <= STRIDED_RANK:
            continue
        src = view.args[0]
        if not isinstance(src, Node):
            continue
        src_shape = list(_val(src).shape)
        view_shape = list(_val(view).shape)
        if len(view_shape) != len(src_shape) + 1 or extent_keys(view_shape[:-2]) != extent_keys(
            src_shape[:-1]
        ):
            continue
        inner = _static_size(view_shape[-1])
        outer = _static_size(view_shape[-2])
        last = _static_size(src_shape[-1])
        if inner is None or outer is None or last is None or outer * inner != last:
            continue
        rewritten = _unbind_consumers(view, split_dim=len(view_shape) - 2)
        if rewritten is None:
            continue
        for squeeze, start, end in rewritten:
            sliced = _insert(
                graph,
                squeeze,
                aten.slice.Tensor,
                (src, len(src_shape) - 1, start * inner, end * inner),
                _val(src),
            )
            _replace(graph, squeeze, sliced)
            stats["split_unbind->slice"] += 1


def _permute_order(node: Any) -> list[int] | None:
    """`permute` の正規化済み軸順（負の添字を畳んだもの）。permute 以外なら None。"""
    if not isinstance(node, Node) or node.op != "call_function":
        return None
    if node.target is not aten.permute.default or len(node.args) != 2:
        return None
    src = node.args[0]
    if not isinstance(src, Node):
        return None
    rank = _rank(src)
    raw = node.args[1]
    if not isinstance(raw, (list, tuple)) or len(raw) != rank:
        return None
    # bool は int の部分型なので明示的に弾く（True が軸 1 に化ける）
    if any(isinstance(axis, bool) or not isinstance(axis, int) for axis in raw):
        return None
    order = [axis % rank for axis in raw]
    return order if sorted(order) == list(range(rank)) else None


def _compose_permute_chains(graph: Graph, stats: Counter) -> None:
    """恒等 `permute` を除去し、隣接する `permute` を 1 回に合成する。

    IR の permute は strided な実体化コピーなので、鎖の各段が 1 dispatch と全要素の
    読み書きを丸ごと 1 回ずつ増やす。`x.permute(p).permute(q)` は
    `z.shape[i] = y.shape[q[i]] = x.shape[p[q[i]]]` なので、合成後の軸順は `p[q[i]]`。

    **中間 permute の他の消費者は壊れない** — 書き換えるのは下流側の入力だけで、中間ノード
    自体は（消費者が残っていれば）そのまま残る。誰も見なくなった中間だけが後段の DCE で
    消える。グラフのノード列は位相順なので、3 段以上の鎖も上流から順に畳まれて 1 パスで
    1 ノードに収束する。
    """
    for permute in list(graph.nodes):
        outer = _permute_order(permute)
        if outer is None:
            continue
        first = permute.args[0]
        if not isinstance(first, Node):
            continue
        if outer == list(range(len(outer))):
            _replace(graph, permute, first)
            stats["permute:identity"] += 1
            continue
        inner = _permute_order(first)
        if inner is None:
            continue
        base = first.args[0]
        if not isinstance(base, Node) or len(inner) != len(outer):
            continue
        composed = [inner[axis] for axis in outer]
        if composed == list(range(len(composed))):
            _replace(graph, permute, base)
            stats["permute_chain:identity"] += 1
            continue
        permute.args = (base, composed)
        stats["permute_chain:composed"] += 1


def _reshape_chain_back(node: Node) -> Node:
    """reshape 系を遡って「要素順が同じ」最上流のテンソルを返す。"""
    current = node
    while _is_reshape(current) and _sole_user(current) is not None:
        src = current.args[0]
        if not isinstance(src, Node):
            break
        current = src
    return current


def _reshape_chain_forward(node: Node) -> Node:
    """clone / reshape 系を下って、要素順が同じままの最下流ノードを返す。"""
    current = node
    while True:
        user = _sole_user(current)
        if user is None or not (_is_reshape(user) or user.target is aten.clone.default):
            return current
        current = user


def _lower_reshape_permute(graph: Graph, stats: Counter) -> None:
    """rank≥5 の `reshape → permute → reshape` を rank4 の隣接転置列へ書き換える。

    DiT の patchify / unpatchify（rank8 の 9 ノード）がこの形。任意の軸置換は隣接互換の積に
    分解でき、隣接互換は「前後をまとめた rank4 の転置」で表せる — 途中の値は全て連続なので
    view で自由に括り直せる（permute は実体化される）。長さ 1 の軸は線形添字に寄与しない
    ので先に落とし、転置回数を減らす。

    MUST: 端点（鎖の最上流・最下流）が rank ≤ STRIDED_RANK に収まらない形は fail loudly。
    """
    for permute in list(graph.nodes):
        if permute.op != "call_function" or permute.target is not aten.permute.default:
            continue
        if max(_rank(permute), _rank(permute.args[0])) <= STRIDED_RANK:
            continue
        base = _reshape_chain_back(permute.args[0])
        sink = _reshape_chain_forward(permute)
        if _rank(base) > STRIDED_RANK or _rank(sink) > STRIDED_RANK:
            raise NotImplementedError(
                f"rank ≤ {STRIDED_RANK} の端点が見つからない reshape/permute 連鎖: "
                f"{base.name}(rank{_rank(base)}) → {permute.name} → "
                f"{sink.name}(rank{_rank(sink)})"
            )
        shape = list(_val(permute.args[0]).shape)
        order = [int(d) % len(shape) for d in permute.args[1]]
        out_shape = list(_val(sink).shape)
        numel = extent_key(_numel(shape))
        if numel != extent_key(_numel(list(_val(base).shape))) or numel != extent_key(
            _numel(out_shape)
        ):
            raise NotImplementedError(f"要素数が食い違う reshape/permute 連鎖: {permute.name}")

        kept = [index for index, size in enumerate(shape) if _static_size(size) != 1]
        position = {axis: slot for slot, axis in enumerate(kept)}
        sizes = [shape[index] for index in kept]
        target = [position[axis] for axis in order if axis in position]

        ref = _val(base)
        current = _as_contiguous(graph, sink, base, ref)
        state = list(range(len(sizes)))
        for slot in range(len(state)):
            index = state.index(target[slot])
            while index > slot:
                head = _numel([sizes[axis] for axis in state[: index - 1]])
                left, right = sizes[state[index - 1]], sizes[state[index]]
                tail = _numel([sizes[axis] for axis in state[index + 1 :]])
                current = _insert(
                    graph, sink, aten.view.default, (current, [head, left, right, tail]), ref
                )
                current = _insert(graph, sink, aten.permute.default, (current, [0, 2, 1, 3]), ref)
                current = _insert_contiguous(graph, sink, current, ref)
                state[index - 1], state[index] = state[index], state[index - 1]
                index -= 1
                stats["reshape_permute:transpose"] += 1
        current = _insert(graph, sink, aten.view.default, (current, out_shape), ref)
        _replace(graph, sink, current)
        stats["reshape_permute"] += 1


def _collect_dead_code(graph: Graph, stats: Counter) -> None:
    """パス群の途中で 1 度 DCE する（**位置に意味がある** — ADR 0016 の移植時の構造差）。

    畳んだ元パターン（pow / mean / rsqrt・ガードの eq/any/logical_not）をここで消さないと、
    後続のパス（`_pow2_to_mul` / `_promote_scalar_operands`）が**死んだ部分木を書き換えて
    統計だけが膨らむ**。IR は変わらないので緑のまま気づけない類の劣化になる。
    """
    before = len(graph.nodes)
    graph.eliminate_dead_code()
    stats["dead_code_collected"] += before - len(graph.nodes)


def _pow2_to_mul(graph: Graph, stats: Counter) -> None:
    """`aten.pow(x, 2)` → `mul(x, x)`（M0 語彙に pow は無いが同値な mul はある）。

    MUST: 指数は **bool でない整数の 2** だけ。`2.0` は `2.0 == 2` で素通りするが、整数
    テンソルに当てると型昇格（i64 → f32）が起きて `mul(x, x)` とは別の値になる。
    MUST: dtype 不変を要求する。昇格を伴う形は恒等な書き換えではないので、触らず未対応 op の
    全件列挙に回す（黙って近似しない）。
    """
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not aten.pow.Tensor_Scalar:
            continue
        exponent = node.args[1]
        if isinstance(exponent, bool) or not isinstance(exponent, int) or exponent != 2:
            continue
        src = node.args[0]
        if not isinstance(src, Node):
            continue
        src_val = src.meta.get("val")
        out_val = node.meta.get("val")
        if not isinstance(src_val, torch.Tensor) or not isinstance(out_val, torch.Tensor):
            continue
        if src_val.dtype is not out_val.dtype:
            continue
        with graph.inserting_before(node):
            replacement = graph.call_function(aten.mul.Tensor, (src, src))
        # convert は全ノードの meta["val"]（FakeTensor）を要求する。
        replacement.meta.update(node.meta)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        stats["pow2->mul"] += 1


def _drop_identity_repeat(graph: Graph, stats: Counter) -> None:
    """`aten.repeat(x, [1, …, 1])` → x。

    MUST: **引数が全て 1 かつ本数 = 入力 rank** のときだけ発火する（recon §6-4）。repeat は
    先頭に軸を足せる（rank-2 に `[1,1,1]` を当てると rank-3 になる）ので、本数を見ないと
    恒等化で shape が黙ってずれる。B > 1 の本物の repeat は IR 語彙に無いため、ここで
    書き換えずに残せば未対応 op の全件列挙に出る（黙って近似しない）。
    """
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not aten.repeat.default:
            continue
        src = node.args[0]
        repeats = node.args[1]
        if not isinstance(src, Node) or not isinstance(repeats, (list, tuple)):
            continue
        # SymInt の repeat 数はここで int と比較しない（guard を誘発する）。素の int だけ見る。
        if not all(isinstance(count, int) and not isinstance(count, bool) for count in repeats):
            continue
        if len(repeats) != src.meta["val"].dim() or any(count != 1 for count in repeats):
            continue
        node.replace_all_uses_with(src)
        graph.erase_node(node)
        stats["drop_identity_repeat"] += 1


def _drop_identity_add(graph: Graph, stats: Counter) -> None:
    """`aten.add(x, 0)` → x（**Python スカラの 0** のときだけ）。

    NOTE: f32 では `x + 0.0` が x = −0.0 のときだけ +0.0 を返す（唯一の差）。IR 語彙には
    **符号付きゼロを区別する op が実在する** — `div` と `sqrt` がそれで、x = −0.0 なら
    `1 / (x + 0.0)` は torch で +∞、書き換え後の `1 / x` は −∞ になる（`sqrt(±0) = ±0` も
    同じ向きの差）。実測 10 ファミリでは消える add の下流が div の分母 / sqrt の引数に
    到達しないため、乖離の可能性を受容してこの書き換えを残している（docs/limitations.md）。
    消費側で −0.0 が出うる形を足すときは、このパスの適用条件を先に見直すこと。
    MUST: 型昇格を伴う形（`add(i64, 0.0)` → f32）は書き換えない。恒等ではなく cast なので、
    畳み込みか未対応 op の列挙に回す。
    """
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in (aten.add.Tensor, aten.add.Scalar):
            continue
        if len(node.args) != 2 or node.kwargs.get("alpha", 1) != 1:
            continue
        src, addend = node.args
        if not isinstance(src, Node) or isinstance(addend, bool):
            continue
        if not isinstance(addend, (int, float)) or addend != 0:
            continue
        if src.meta["val"].dtype is not node.meta["val"].dtype:
            continue
        node.replace_all_uses_with(src)
        graph.erase_node(node)
        stats["drop_identity_add"] += 1


#: スカラ被演算子を持ちうる二項 op → (置き換え先, スカラが左辺か)。
#:
#: `rsub.Scalar(x, s)` だけは引数の並びと意味が逆（`s − x`）なので明示的に左辺へ回す。
#: それ以外は「Node でない側がスカラ」で並びを判定する — 実測は `sub.Tensor(1, mask)` の
#: ように **.Tensor overload に Python スカラが左で入る**形も出るため（recon の mask 経路）。
_SCALAR_BINARY: dict[Any, tuple[Any, bool]] = {
    aten.add.Tensor: (aten.add.Tensor, False),
    aten.add.Scalar: (aten.add.Tensor, False),
    aten.sub.Tensor: (aten.sub.Tensor, False),
    aten.sub.Scalar: (aten.sub.Tensor, False),
    aten.mul.Tensor: (aten.mul.Tensor, False),
    aten.mul.Scalar: (aten.mul.Tensor, False),
    aten.div.Tensor: (aten.div.Tensor, False),
    aten.div.Scalar: (aten.div.Tensor, False),
    aten.rsub.Scalar: (aten.sub.Tensor, True),
}


def _promote_scalar_operands(graph: Graph, stats: Counter) -> None:
    """`add/sub/mul/div(tensor, スカラ)` → 二項 op + **rank-1 定数**（broadcast で吸収）。

    定数は `aten.full.default([1], value, dtype=…)` として挿入する。convert の畳み込み
    allowlist に載っている op なので、消費側が畳み込み対象なら定数ごと畳まれ、そうでなければ
    initializer になる（i32 なら i32 initializer — ADR 0010）。

    MUST: 定数の dtype はテンソル側と揃え、**型昇格を伴う形は書き換えない**（`div(i64, 128)`
    は f32 を返す = cast を含む）。揃えずに rank-1 定数へ落とすと、IR の二項 op が要求する
    「入力 dtype は同型」を満たさないまま黙って別の値になる。
    MUST: `alpha != 1` は触らない（IR 語彙に係数付き加算は無い — convert が落とす）。
    MUST: 非有限のスカラは触らない（IR v1 は非有限値を値レベルで拒否する）。
    """
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        entry = _SCALAR_BINARY.get(node.target)
        if entry is None or len(node.args) != 2:
            continue
        target, scalar_is_left = entry
        if node.kwargs.get("alpha", 1) != 1 or set(node.kwargs) - {"alpha"}:
            continue
        left, right = node.args
        # MUST: 定数を「元のスカラが居た側」に置く。sub / div は非可換で、実測には
        # `sub.Tensor(1, mask)` のようにスカラが**左**に来る形がある（recon の mask 経路）—
        # 並びを固定すると符号が黙って反転する。
        if scalar_is_left:
            src, scalar, const_first = left, right, True
        elif isinstance(left, Node) and not isinstance(right, Node):
            src, scalar, const_first = left, right, False
        elif isinstance(right, Node) and not isinstance(left, Node):
            src, scalar, const_first = right, left, True
        else:
            continue
        if not isinstance(src, Node) or isinstance(scalar, bool):
            continue
        if not isinstance(scalar, (int, float)) or not math.isfinite(scalar):
            continue
        src_val = src.meta.get("val")
        out_val = node.meta.get("val")
        if not isinstance(src_val, torch.Tensor) or not isinstance(out_val, torch.Tensor):
            continue
        if src_val.dtype is not out_val.dtype:
            continue
        # MUST: rank-0（スカラテンソル）は昇格させない。rank-1 定数との broadcast は
        # `[] × [1] → [1]` で **rank が上がる** — meta は元の `[]` のままなので、宣言と実体が
        # 食い違ったグラフが黙って出来る。rank ≥ 1 なら `[1]` は必ず吸収される。
        if src_val.dim() < 1:
            continue
        with graph.inserting_before(node):
            const = graph.call_function(aten.full.default, ([1], scalar), {"dtype": src_val.dtype})
        const.meta["val"] = src_val.new_full((1,), scalar)
        args = (const, src) if const_first else (src, const)
        with graph.inserting_before(node):
            replacement = graph.call_function(target, args)
        replacement.meta.update(node.meta)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        stats["promote_scalar_operand"] += 1


def _eq_zero_to_not_bool(graph: Graph, stats: Counter) -> None:
    """`aten.eq(x, 0)` → `bitwise_not(_to_copy(x, bool))`（ADR 0015）。

    `scores.masked_fill(attn_mask == 0, -1e4)` の形（recon §2 の enc_p / flow 行）を、
    新しい比較 op を足さずに既存語彙へ落とす。同値性は cast 規約「x → bool は x != 0」
    からの帰結: `not (x != 0)` は `x == 0` そのもの（NaN でも符号付きゼロでも一致する —
    `NaN != 0` は真なので否定して偽、`-0.0 != 0` は偽なので否定して真）。

    MUST: 右辺が **0**（bool でない数値）のときだけ発火する。他の値には成り立たない書き換え
    なので、触らずに未対応 op の全件列挙へ回す（黙って近似しない）。
    MUST: 出力が bool であることを要求する。torch の eq は常に bool を返すが、meta が別の
    dtype を持つ形（型昇格を伴う書き換え）を素通しにすると宣言と実体が食い違う。
    """
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in (aten.eq.Scalar, aten.eq.Tensor):
            continue
        if len(node.args) != 2 or node.kwargs:
            continue
        src, other = node.args
        if not isinstance(src, Node) or isinstance(other, bool):
            continue
        if not isinstance(other, (int, float)) or other != 0:
            continue
        src_val = src.meta.get("val")
        out_val = node.meta.get("val")
        if not isinstance(src_val, torch.Tensor) or not isinstance(out_val, torch.Tensor):
            continue
        if out_val.dtype is not torch.bool:
            continue
        with graph.inserting_before(node):
            truthy = graph.call_function(aten._to_copy.default, (src,), {"dtype": torch.bool})
            replacement = graph.call_function(aten.bitwise_not.default, (truthy,))
        # convert は全ノードの meta["val"]（FakeTensor）を要求する。cast の出力は
        # 「x != 0」なので元テンソルと同形の bool。
        truthy.meta.update(node.meta)
        replacement.meta.update(node.meta)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        stats["eq_zero->bitwise_not_cast"] += 1


def _select_to_squeeze(graph: Graph, stats: Counter) -> None:
    """`select.int(長さ 1 の軸, 添字 0)` → `squeeze`（汎用衛生 — ADR 0016）。

    MUST: **記号次元の軸は fail loudly**。長さを export 時のヒント値で採ると「ヒント値 1 の
    記号次元」が squeeze に化け、実行時（ヒント以外の束縛）で無言の誤値になる — 静かに
    間違える形なので、ここで止める以外に検出器が無い。
    NOTE: 静的で長さ 2 以上の軸は書き換えずに残す（`aten.select.int` として未対応 op の
    全件列挙に出る）。こちらは黙って間違える形ではないので、診断は 1 度に全部並べる側
    （ADR 0005）に寄せる。
    """
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not aten.select.int:
            continue
        src = node.args[0]
        if not isinstance(src, Node):
            continue
        src_val = _val(src)
        rank = src_val.dim()
        raw_dim, index = int(node.args[1]), int(node.args[2])
        if not -rank <= raw_dim < rank:
            continue
        dim = raw_dim % rank
        size = _static_size(src_val.shape[dim])
        if size is None:
            raise NotImplementedError(
                f"記号次元の軸を切る select は未対応: {node.name} dim={raw_dim} index={index}"
                "（ヒント値で長さを採ると実行時に無言の誤値になる）"
            )
        if index != 0 or size != 1:
            continue
        with graph.inserting_before(node):
            replacement = graph.call_function(aten.squeeze.dims, (src, [dim]))
        replacement.meta.update(node.meta)
        _replace(graph, node, replacement)
        stats["select->squeeze"] += 1


def _split_to_slices(graph: Graph, stats: Counter) -> None:
    """`split_with_sizes(x, sizes, dim)` + `getitem` → `slice` の列（ADR 0014）。

    IR に多出力 op は無い（ノードは単一出力 — docs/ir-v1.md）ので、分割は取り出し口ごとの
    slice に開く。実測は ConvFlow / ResidualCoupling の `torch.split(x, [half]*2, 1)`
    （recon §2）で、消費側は必ず `getitem` の定数添字。

    MUST: **消費者が getitem だけ**の形にしか発火しない。タプルそのものを他所へ渡す形は
    書き換えが同値でないので、触らず未対応 op の全件列挙に回す（黙って近似しない）。
    MUST: 分割軸は静的（記号長の分割は sizes の和と軸長の対応が宣言だけでは決められない）。
    MUST: sizes の和が軸長ちょうどであることを見る。torch は端数を許さないが、ここを見ずに
    slice へ開くと「最後の区間が足りない / はみ出す」形が黙って通る。

    NOTE: 「消費者が getitem 以外」「分割軸が記号」「sizes の和が合わない」の 3 分岐は
    torch.export の出力からは到達しない（FX は必ず getitem で開き、静的 sizes の分割は軸を
    特殊化し、端数は torch 自身が拒否する）。防御として残すが、テストは書けない
    （書ける形が無い）。
    """
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not aten.split_with_sizes.default:
            continue
        src = node.args[0]
        sizes = node.args[1]
        if not isinstance(src, Node) or not isinstance(sizes, (list, tuple)):
            continue
        if any(isinstance(size, bool) or not isinstance(size, int) for size in sizes):
            continue
        src_val = src.meta.get("val")
        if not isinstance(src_val, torch.Tensor):
            continue
        rank = src_val.dim()
        raw_dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)
        if isinstance(raw_dim, bool) or not isinstance(raw_dim, int) or not -rank <= raw_dim < rank:
            continue
        dim = raw_dim % rank
        extent = src_val.shape[dim]
        if isinstance(extent, torch.SymInt) and extent.node.expr.free_symbols:
            continue
        if sum(sizes) != int(extent):
            continue
        users = list(node.users)
        if not users or any(user.target is not operator.getitem for user in users):
            continue
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        replaced = 0
        for user in users:
            index = user.args[1]
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(sizes):
                continue
            with graph.inserting_before(user):
                sliced = graph.call_function(
                    aten.slice.Tensor, (src, dim, offsets[index], offsets[index + 1])
                )
            # convert は全ノードの meta["val"]（FakeTensor）を要求する。getitem の meta は
            # 取り出した 1 本ぶんそのもの。
            sliced.meta.update(user.meta)
            user.replace_all_uses_with(sliced)
            graph.erase_node(user)
            replaced += 1
        if replaced != len(users):
            # 添字が読めない getitem が残った = 分割ノードを消せない（部分的な書き換えは
            # 残った getitem が壊れた参照になるので、ここまでの slice ごと DCE に任せる）。
            continue
        graph.erase_node(node)
        stats["split_with_sizes->slice"] += replaced


def _passes(placeholders: _Placeholders) -> tuple[Callable[[Graph, Counter], None], ...]:
    """走らせるパスを順に並べる。

    MUST: `_fold_rms_norm` → `_pow2_to_mul` / `_promote_scalar_operands` の順、
    `_drop_identity_add` → `_promote_scalar_operands` の順（モジュール docstring 参照）。
    `_eq_zero_to_not_bool` は後ろ — 先に走らせても結果は同じだが、スカラ昇格が
    `eq(x, 0)` を二項 op として掴む形が将来入ったときに順序依存を作らないため。
    """
    return (
        _drop_metadata_asserts,
        # 高位パターンの畳み込み・除去（元パターンを他のパスが崩す前に走らせる）
        _fold_rms_norm,
        partial(_drop_safe_softmax_guard, placeholders=placeholders),
        _additive_attn_mask,
        # rank 下げ（発火は rank > STRIDED_RANK 限定）
        _lower_unit_expand,
        _lower_split_unbind,
        _lower_reshape_permute,
        # rank 下げが挟む contiguous を跨がないので、位相順に 1 度掃くだけで鎖は畳み切れる
        _compose_permute_chains,
        # MUST: ここで 1 度 DCE する（位置に意味がある — _collect_dead_code の docstring）
        _collect_dead_code,
        _pow2_to_mul,
        _drop_identity_repeat,
        _drop_identity_add,
        _promote_scalar_operands,
        _eq_zero_to_not_bool,
        _select_to_squeeze,
        # 順序自由（split の入出力は他のパスが触らない形）。末尾に置くのは、生成した slice を
        # 他のパスが再訪しないと分かるようにするため。
        _split_to_slices,
    )


def normalize_graph(ep: ExportedProgram) -> dict[str, int]:
    """FX グラフを in-place で正規化し、パスごとの発火回数を返す。

    MUST: 受け取るのは GraphModule ではなく **ExportedProgram**。safe-softmax ガードの
    不活性証明は持ち上げられた定数 / buffer の**実値**を要る（{@link _Placeholders}）。
    """
    gm = ep.graph_module
    graph = gm.graph
    stats: Counter[str] = Counter()
    placeholders = _Placeholders(ep)
    for run in _passes(placeholders):
        run(graph, stats)
    graph.eliminate_dead_code()
    graph.lint()
    gm.recompile()
    return dict(stats)
