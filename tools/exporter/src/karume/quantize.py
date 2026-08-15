"""重みの fake-quant（格納 dtype で表現可能な値へ丸める）— ADR 0006 / 0018 / 0019。

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

from collections.abc import Mapping
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


def fake_quant_int8(model: nn.Module) -> Int8Report:
    """重みスロットを持つモジュールの重みを per-channel symmetric int8 の表現可能値へ丸める。

    対象は {@link QUANT_CHANNEL_AXES} に載る型のモジュールの `weight` だけ — bias も norm 系
    weight も**触らない**（bias は常に f32 が ADR 0006 の根治形で、量子化の対象に載せた瞬間に
    プロトタイプの降格バグが再来する）。

    MUST: 呼ぶ順序は「実効重みが確定した後・参照/golden の採取より前」（モジュール docstring）。

    戻り値の scale 台帳は emit へそのまま渡す。1 本も量子化できなかった場合は fail loudly —
    「`--dtype i8` を指定したのに実質 f32 で書けてしまった」を沈黙させないため（ADR 0006）。
    """
    scales: dict[str, torch.Tensor] = {}
    elements = 0
    with torch.no_grad():
        for name, module in model.named_modules():
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
