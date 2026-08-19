"""重みの fake-quant（格納 dtype で表現可能な値へ丸める）— ADR 0006 / 0018 / 0019 / 0069。

意味論は「格納のみ量子化・計算は f32」。エクスポータが重みを**先に**丸めてから参照と
golden を採ることで、GPU 側「圧縮格納 + f32 計算」と torch 側「丸め済み重みの f32 計算」が
**同じ数**を計算する対応になり、量子化誤差と実装誤差が分離される。

MUST: 丸めは**参照・golden の採取より前**に当てる（ADR 0006）。後に当てると参照だけが
元の重みで計算され、E2E の差が「量子化誤差 + 実装誤差」の合成になって tolerance の意味が
失われる（そして tolerance を緩める方向にしか作用しないので、緑のまま検出力だけが落ちる）。

MUST: 丸めは**実効重み**に当てる — LoRA の焼き込み・weight_norm の除去・CausalConv3d の
時間方向スライスのように重みを書き換える処理は丸めより前に済ませる。f16 では「合成後の値が
f16 表現可能でなくなる」ため、i8 では**さらに strict** で、捨てられる要素が amax に効くと
per-channel scale そのものがずれる（値は全要素で変わる）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import nn


class QuantizeError(ValueError):
    """丸めが表現可能値へ落とせなかった / 量子化対象の素性が想定と違う。"""


@dataclass(frozen=True)
class RoundReport:
    """丸めた本数と要素数（「f16 指定なのに 0 本」を沈黙させないための計数）。"""

    parameters: int
    buffers: int
    elements: int

    def describe(self) -> str:
        return f"parameters {self.parameters} / buffers {self.buffers} / {self.elements:,} elements"


def round_weights_to_f16(model: torch.nn.Module) -> RoundReport:
    """パラメータと f32 バッファを f16 表現可能値へ丸める（計算は f32 のまま）。

    対象を「パラメータ + f32 バッファ」にするのは、格納 f16 になりうる initializer が
    そこからしか来ないから（畳み込み定数と焼き込み定数はグラフ意味論の一部で、golden が
    元値で計算されるため f32 のまま格納する — 適格判定は emit.py 側）。

    MUST: 有限値が非有限へ飽和したら fail loudly。`Tensor.to(float16)` は f16 の値域
    （|x| ≤ 65504）を超えた要素を静かに ±inf にするため、無検査だと「重みに inf が入った
    モデル」と「同じ inf で計算した参照」が**一致してしまい**、E2E が緑のまま出力が壊れる。
    """
    counts = {"parameter": 0, "buffer": 0}
    elements = 0
    with torch.no_grad():
        # named_parameters / named_buffers は既定で重複（重み共有）を 1 度しか出さない。
        # f16 丸めは冪等なので 2 度当たっても値は変わらないが、計数は実体の本数で出す。
        targets = [("parameter", name, t) for name, t in model.named_parameters()]
        targets += [("buffer", name, t) for name, t in model.named_buffers()]
        for kind, name, tensor in targets:
            if tensor.dtype is not torch.float32:
                continue
            rounded = tensor.to(torch.float16).to(torch.float32)
            if torch.isfinite(tensor).all() and not torch.isfinite(rounded).all():
                overflow = int((~torch.isfinite(rounded)).sum())
                raise QuantizeError(
                    f"'{name}': f16 への丸めで有限値 {overflow} 個が非有限へ飽和した"
                    f"（f16 の値域は |x| ≤ 65504・実測 max |x| = "
                    f"{float(tensor.abs().max()):.4g}）"
                )
            tensor.copy_(rounded)
            counts[kind] += 1
            elements += tensor.numel()
    return RoundReport(parameters=counts["parameter"], buffers=counts["buffer"], elements=elements)


# ---- i8（per-channel symmetric）— ADR 0019 ---------------------------------

#: 量子化幅の片側（**−128 は使わない**）。±127 に閉じると最大絶対値要素が `q = ±127` に
#: 乗って `q·scale` で厳密に復元されるので、fake-quant が**冪等**になる（再適用でビット不変
#: — `tests/test_quantize.py` が固定する）。−128 を許すと scale だけが動く再量子化が起きる。
INT8_MAX = 127

#: per-channel scale の**チャネル軸**（モジュール型 → 重みテンソルの軸番号 — ADR 0019）。
#: 出力チャネルの軸で、`ConvTranspose1d` だけ重みが `[Cin, Cout, K]` の転置レイアウトなので 1。
#:
#: MUST: `ops.WEIGHT_CHANNEL_AXES`（op 名で引く同じ表）と**同じ軸**を返す。片方だけ動かすと
#: 「emit した scale の軸」と「カーネルが引く軸」が食い違い、値が黙って壊れる。対応は
#: `tests/test_quantize.py` の突合テストが固定する（モジュール型 → op 名の対応表つき）。
QUANT_CHANNEL_AXES: Mapping[type[nn.Module], int] = MappingProxyType(
    {
        nn.Linear: 0,
        nn.Conv1d: 0,
        nn.Conv2d: 0,
        nn.ConvTranspose1d: 1,
        nn.Embedding: 0,
    }
)

#: 量子化対象にできるモジュール型の一覧（{@link QUANT_CHANNEL_AXES} の鍵から導出）。
#: `op_types` へ「i8 と同じ 5 種」を渡すときの綴り — 表を写した別リストにはしない
#: （独立に動くと「軸を引ける型」と「対象にできる型」が黙って割れる）。
QUANT_MODULE_TYPES: tuple[type[nn.Module], ...] = tuple(QUANT_CHANNEL_AXES)


@dataclass(frozen=True)
class Int8Report:
    """量子化した重みの scale 台帳と計数。

    `scales` のキーは**モデル内 FQN**（`<module>.weight`）で、`convert.py` が safetensors の
    テンソルキーに使う FQN と同じ空間。`id(tensor)` で突き合わせない（ADR 0006）— パラメータ
    同一性は正規化・焼き込みで簡単に崩れ、崩れても黙って「対象 0 本」に落ちるだけだから。
    """

    #: FQN → scale（重みと同 rank の keepdim 形・F32）。
    scales: Mapping[str, torch.Tensor]
    modules: int
    elements: int

    def describe(self) -> str:
        return f"modules {self.modules} / {self.elements:,} elements"


def _channel_axis(module: nn.Module) -> int | None:
    """モジュールの per-channel 軸（対象外なら None）。

    完全一致を先に見てから isinstance へ落とす — 実モデルは `nn.Linear` の薄い派生を使う
    ことがあり、型の完全一致だけだと**黙って対象から外れる**（emit 側の適格判定は消費 op で
    決まるので、外れた重みは「適格なのに scale が無い」として fail loudly になるが、
    ここで拾えるものは拾う）。2 つ以上に当たる型は軸が決まらないので落とす。
    """
    exact = QUANT_CHANNEL_AXES.get(type(module))
    if exact is not None:
        return exact
    axes = {axis for cls, axis in QUANT_CHANNEL_AXES.items() if isinstance(module, cls)}
    if len(axes) > 1:
        raise QuantizeError(
            f"{type(module).__name__}: per-channel 軸が 1 つに決まらない（候補 {sorted(axes)}）"
        )
    return next(iter(axes), None)


def channel_scale(weight: torch.Tensor, axis: int) -> torch.Tensor:
    """`scale = clamp(amax / 127, f32 tiny)` を weight と同 rank の keepdim 形で返す。

    下限 clamp が要るのは全ゼロチャネル（`amax == 0`）— そのまま割ると NaN になる。
    clamp 後は `q = 0` に落ちて `q·scale = 0` が厳密に元値へ戻る。
    """
    reduced = [dim for dim in range(weight.dim()) if dim != axis]
    amax = weight.abs().amax(dim=reduced, keepdim=True) if reduced else weight.abs()
    return torch.clamp(amax / INT8_MAX, min=torch.finfo(torch.float32).tiny)


def quantize_to_int8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """`q = clamp(round(w / scale), ±127)` を int8 で返す（scale は**再計算しない**）。

    MUST: 呼び出し側は fake-quant が使った scale を**そのまま**渡す。ここで amax から
    引き直すと f32 の丸めで `scale` が 1ulp 動きうるので、格納値が golden を採ったときの
    重みと一致しなくなる（ADR 0019）。
    """
    return torch.round(weight / scale).clamp_(-INT8_MAX, INT8_MAX).to(torch.int8)


def fake_quant_int8(model: nn.Module, include: Callable[[str], bool] | None = None) -> Int8Report:
    """重みスロットを持つモジュールの重みを per-channel symmetric int8 の表現可能値へ丸める。

    対象は {@link QUANT_CHANNEL_AXES} に載る型のモジュールの `weight` だけ — bias も norm 系
    weight も**触らない**（bias は常に f32 が ADR 0006 の根治形で、量子化の対象に載せた瞬間に
    プロトタイプの降格バグが再来する）。

    `include` は**モジュール FQN**（`named_modules` の名前）の述語（None = 全対象）。混成格納
    （例: embedding は i8・linear は i4）で対象を割るための口。MUST: 混成では i8 / i4 の対象を
    **排他に**割る — tied 重み（lm_head = embed_tokens）を両方に通すと二重丸めになり、後に
    走った側の丸めだけが値に残って先の scale 台帳が実値と食い違う。

    MUST: 呼ぶ順序は「実効重みが確定した後・参照/golden の採取より前」（モジュール docstring）。

    戻り値の scale 台帳は emit へそのまま渡す。1 本も量子化できなかった場合は fail loudly —
    「`--dtype i8` を指定したのに実質 f32 で書けてしまった」を沈黙させないため（ADR 0006・
    `include` が全対象を落とした場合も同じ）。
    """
    scales: dict[str, torch.Tensor] = {}
    elements = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            # include の判定は _channel_axis より先 — 対象外モジュールの型曖昧性
            # （複数型ヒット）で落とさない。
            if include is not None and not include(name):
                continue
            axis = _channel_axis(module)
            if axis is None:
                continue
            weight = getattr(module, "weight", None)
            if weight is None:
                continue
            fqn = f"{name}.weight" if name else "weight"
            if weight.dtype is not torch.float32:
                raise QuantizeError(
                    f"'{fqn}': dtype {weight.dtype} は量子化できない（意味論 f32 のみ）"
                )
            if axis >= weight.dim():
                raise QuantizeError(
                    f"'{fqn}': チャネル軸 {axis} が rank {weight.dim()} の重みに無い"
                )
            scale = channel_scale(weight, axis)
            weight.copy_(quantize_to_int8(weight, scale).to(torch.float32) * scale)
            scales[fqn] = scale
            elements += weight.numel()
    if not scales:
        raise QuantizeError(
            "格納 i8 を指定したが per-channel 量子化できる重みが 1 本も無い"
            f"（対象の型: {', '.join(cls.__name__ for cls in QUANT_CHANNEL_AXES)}）"
        )
    return Int8Report(scales=scales, modules=len(scales), elements=elements)


# ---- 対象選択と「in 軸」の一般化（丸め方式を問わない共有部）-----------------


def iter_quant_targets(
    model: nn.Module,
    op_types: tuple[type[nn.Module], ...] = (nn.Linear,),
    include: Callable[[str], bool] | None = None,
) -> Iterator[tuple[str, torch.Tensor, int]]:
    """量子化対象の重みを `(FQN, weight, per-channel 軸)` で列挙する（丸め方は問わない）。

    対象選択の**正本** — group 量子化（{@link fake_quant_int4}）も測定専用の方式
    （`quant_methods`）もここを通す。二重実装にすると「出荷する重みの対象」と「品質を
    測った対象」が黙って割れ、測った劣化と出した資産が別物になる。

    `op_types` は `isinstance` で当てる（実モデルは `nn.Linear` の薄い派生を使う — 派生型を
    直に渡すこともできる）。軸は {@link QUANT_CHANNEL_AXES} が正本で、**軸を引けない型は
    素通り**する（`op_types` が丸ごと的外れだった場合は呼び出し側の「対象 0 本」診断で
    出る — 文言は方式ごとに違うのでここでは判定しない）。

    `include` は**モジュール FQN** の述語（None = 全対象）。判定を型より先に置くのは
    {@link fake_quant_int8} と同じ理由 — 対象外モジュールの型曖昧性で落とさないため。
    """
    for name, module in model.named_modules():
        if include is not None and not include(name):
            continue
        if not isinstance(module, op_types):
            continue
        axis = _channel_axis(module)
        if axis is None:
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        fqn = f"{name}.weight" if name else "weight"
        if weight.dtype is not torch.float32:
            raise QuantizeError(
                f"'{fqn}': dtype {weight.dtype} は量子化できない（意味論 f32 のみ）"
            )
        if axis >= weight.dim():
            raise QuantizeError(f"'{fqn}': チャネル軸 {axis} が rank {weight.dim()} の重みに無い")
        yield fqn, weight, axis


def channel_rows(weight: torch.Tensor, axis: int) -> torch.Tensor:
    """重みを `[チャネル, in]` の 2 次元へ畳む（group の軸 = 各 op の「in 軸」一般化）。

    - linear `[O,I]` / embedding `[V,D]` … 恒等（既に `[チャネル, in]` の形）
    - conv1d `[Cout,Cin,K]` / conv2d `[Cout,Cin,Kh,Kw]` … 出力チャネルごとに受容野を平坦化
      （`Cin·K` / `Cin·Kh·Kw`）
    - conv_transpose1d `[Cin,Cout,K]` … 転置レイアウトなので軸 1 を先頭へ回してから平坦化
      （i8 の per-channel 軸と同じ向き — {@link QUANT_CHANNEL_AXES}）

    NOTE: 軸 0 の連続重みでは **view** が返る（コピーしない）が、`ConvTranspose1d` は
    `movedim` で非連続になるため `reshape` が**コピー**を作る。書き戻しは必ず
    {@link restore_channel_rows} を通す — 平坦形へ直接書いても元の重みには反映されない。
    """
    return weight.movedim(axis, 0).reshape(int(weight.shape[axis]), -1)


def restore_channel_rows(rows: torch.Tensor, weight: torch.Tensor, axis: int) -> torch.Tensor:
    """{@link channel_rows} の逆 — `[チャネル, in]` を `weight` の形へ戻す。"""
    moved = list(weight.shape)
    moved.insert(0, moved.pop(axis))
    return rows.reshape(moved).movedim(0, axis)


# ---- i4（K 方向 group symmetric）— ADR 0069 ---------------------------------

#: 量子化幅の片側（**−8 は使わない**）。±7 に閉じると group の amax 要素が `q = ±7` に乗って
#: `q·scale` で厳密に復元されるので、fake-quant が**冪等**になる（ADR 0019 の ±127 論証の
#: 4bit 版 — ADR 0069 決定 3）。−8 を許すと scale だけが動く再量子化が起きる。
INT4_MAX = 7

#: 既定の group 長（ADR 0069 追記 = Phase 0 sweep の実測で確定した 32）。格納欄なので、
#: サイズ優先の資産が 64 / 128 を選ぶことは妨げない（受理集合は 2 冪かつ 16 以上 — 値域の
#: 検査は格納側の規則として `verify.py` が持つ）。
DEFAULT_GROUP_SIZE = 32


@dataclass(frozen=True)
class Int4Report:
    """group 対称 int4 で丸めた重みの scale 台帳と計数（`Int8Report` と同じ器）。

    `scales` のキーは i8 と同じ**モデル内 FQN**（`<module>.weight`）。値は
    {@link channel_rows} で畳んだ形の group scale = `[チャネル, group 数]` の F32。

    - linear `W[O,I]` → `[O, I/group_size]` / embedding `W[V,D]` → `[V, D/group_size]` …
      **重みと同 rank・最終次元だけ group 数**（ADR 0069 決定 3 の形 = emit が受ける形。
      i8 の keepdim broadcast 形とは受理集合が交わらない別の形）
    - conv 系 … 受容野を平坦化した rank 2 で、**重みと rank が合わない**（emit へは渡せない
      — {@link fake_quant_int4} の docstring）
    """

    #: FQN → scale（group 形 `[チャネル, group 数]`・F32）。
    scales: Mapping[str, torch.Tensor]
    group_size: int
    modules: int
    elements: int

    def describe(self) -> str:
        return f"modules {self.modules} / {self.elements:,} elements / group {self.group_size}"


def grouped_view(weight: torch.Tensor, group_size: int, where: str) -> torch.Tensor:
    """量子化軸（**最終次元 = in 軸**）を `[…, groups, group_size]` へ割る。

    linear の重みはそのまま最終次元が in 軸で、他の op は {@link channel_rows} で
    `[チャネル, in]` へ畳んでから渡す。

    MUST: 割り切れない形は fail loudly（ADR 0069 決定 2）。端数 group を許すと最後の group
    だけ scale の担当範囲が短くなり、行境界が語境界からずれて平坦添字の展開が黙って別の値を
    出す — 端数を作らない制約で整列問題そのものを消すのが格納側の設計。
    """
    in_axis = int(weight.shape[-1])
    if in_axis % group_size:
        raise QuantizeError(
            f"{where}: 量子化軸（最終次元）{in_axis} が group_size {group_size} で"
            "割り切れない（i4 は端数 group を作らない MUST — ADR 0069 決定 2）"
        )
    return weight.reshape(*weight.shape[:-1], in_axis // group_size, group_size)


def group_size_of(weight: torch.Tensor, scale: torch.Tensor) -> int:
    """group 形の scale から group 長を引く（`[…, groups]` → `in / groups`）。

    MUST: group 長の源は**渡された scale** だけにする。別引数で受け取る形にすると
    「fake-quant が使った scale」と「格納で宣言する group 長」が独立に動けてしまい、
    食い違っても形と型は合うので沈黙誤値になる（ADR 0069 決定 3 の「scale 再計算禁止」と
    同じ穴）。格納側の宣言もここから引く（`emit._plan_i4`）。
    """
    if scale.dim() != weight.dim() or list(scale.shape[:-1]) != list(weight.shape[:-1]):
        raise QuantizeError(
            f"scale {list(scale.shape)} が重み {list(weight.shape)} の group 形"
            "（同 rank・最終次元だけ group 数）でない"
        )
    groups = int(scale.shape[-1])
    in_axis = int(weight.shape[-1])
    if groups < 1 or in_axis % groups:
        raise QuantizeError(f"scale の group 数 {groups} が重みの量子化軸 {in_axis} を割り切らない")
    return in_axis // groups


def group_scale(weight: torch.Tensor, group_size: int, where: str = "重み") -> torch.Tensor:
    """`scale = clamp(amax_group / 7, f32 tiny)` を group 形（`[…, in/group_size]`）で返す。

    下限 clamp が要るのは全ゼロ group（`amax == 0`）— そのまま割ると NaN になる。clamp 後は
    `q = 0` に落ちて `q·scale = 0` が厳密に元値へ戻る（`channel_scale` と同文の 4bit 版）。
    """
    grouped = grouped_view(weight, group_size, where)
    amax = grouped.abs().amax(dim=-1)
    return torch.clamp(amax / INT4_MAX, min=torch.finfo(torch.float32).tiny)


def quantize_to_int4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """`q = clamp(round(w / scale), ±7)` を int8 の器・重みと同じ論理形で返す。

    MUST: 呼び出し側は fake-quant が使った scale を**そのまま**渡す（`quantize_to_int8` と
    同文 — ここで amax から引き直すと f32 の丸めで 1ulp 動きうる）。torch に 4bit の器は
    無いので値は int8 に載せる。**1 バイトへ 2 要素を詰めるのは格納側**（`emit.pack_int4`）で、
    ここは pack 順を知らない。
    """
    grouped = grouped_view(weight, group_size_of(weight, scale), "重み")
    quantized = torch.round(grouped / scale.unsqueeze(-1)).clamp_(-INT4_MAX, INT4_MAX)
    return quantized.reshape(weight.shape).to(torch.int8)


def dequantize_int4(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """`q·scale` を f32・重みと同じ論理形で返す（格納の逆変換）。

    fake-quant（下）と emit の逆変換ビット一致門（ADR 0069 決定 4 ③）が**同じ経路**を通る
    ための共通実装 — 別実装にすると「丸めに使った式」と「格納を検算する式」が独立に動く。
    """
    grouped = grouped_view(quantized.to(torch.float32), group_size_of(quantized, scale), "量子化値")
    return (grouped * scale.unsqueeze(-1)).reshape(quantized.shape)


def fake_quant_int4(
    model: nn.Module,
    group_size: int = DEFAULT_GROUP_SIZE,
    include: Callable[[str], bool] | None = None,
    op_types: tuple[type[nn.Module], ...] = (nn.Linear,),
) -> Int4Report:
    """重みを「in 軸」方向の group symmetric int4 の表現可能値へ丸める（ADR 0069）。

    既定の対象が **`nn.Linear` の `weight` だけ**なのは、i4 の実行経路が linear の重み
    スロット限定で始まるから（ADR 0069 決定 5）。`op_types` は**明示 opt-in の口**で、i8 の
    {@link QUANT_MODULE_TYPES}（全 5 種）まで広げられる — ただし広げてよいのは**品質測定**
    まで。bias も norm 系 weight も触らないのは全対象で同じ。

    MUST: **emit へ渡せるのは linear 分の scale だけ**（`emit._plan_i4` の適格は消費 op が
    linear であること）。conv 系は {@link channel_rows} で受容野を平坦化した rank 2 の scale に
    なり、そもそも重みと rank が合わない。embedding は形こそ合うが実行経路が無いので、
    渡すと emit 側が fail loudly する。広げた対象は「その方式ならどこまで戻るか」を測る側の
    ためのもので、出荷経路ではない（測定専用の方式群は `quant_methods`）。

    group の軸は各 op の「in 軸」を一般化する（{@link channel_rows}）— linear は in 軸、
    conv は出力チャネルごとの受容野（`Cin·K` / `Cin·Kh·Kw`）、`ConvTranspose1d` は転置
    レイアウトの軸 1 ごとに `Cin·K`、embedding は語彙エントリごとに D 軸。

    `include` は**モジュール FQN** の述語（None = 全対象）— 意味と排他 MUST は
    {@link fake_quant_int8} と同文。tied lm_head（= embed_tokens で embedding 側も消費）は
    i4 不適格（emit の linear 限定）なので、混成では include で i4 から外して i8 側へ割る。

    MUST: 呼ぶ順序は「実効重みが確定した後・参照/golden の採取より前」（モジュール docstring）。

    戻り値の scale 台帳は emit へそのまま渡す。1 本も量子化できなかった場合は fail loudly —
    「`--dtype i4` を指定したのに実質 f32 で書けてしまった」を沈黙させないため（ADR 0006・
    `include` が全対象を落とした場合も同じ）。
    """
    scales: dict[str, torch.Tensor] = {}
    elements = 0
    with torch.no_grad():
        for fqn, weight, axis in iter_quant_targets(model, op_types, include):
            rows = channel_rows(weight, axis)
            scale = group_scale(rows, group_size, f"'{fqn}'")
            rounded = dequantize_int4(quantize_to_int4(rows, scale), scale)
            weight.copy_(restore_channel_rows(rounded, weight, axis))
            scales[fqn] = scale
            elements += weight.numel()
    if not scales:
        raise QuantizeError(
            "格納 i4 を指定したが group 量子化できる重みが 1 本も無い"
            f"（対象の型: {', '.join(cls.__name__ for cls in op_types)} — 既定は"
            " nn.Linear のみ・ADR 0069 決定 5）"
        )
    return Int4Report(scales=scales, group_size=group_size, modules=len(scales), elements=elements)
