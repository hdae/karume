"""活性の **per-token symmetric int8** fake-quant（w8a8 実行経路の torch 鏡像）。

ランタイム側の正本は `packages/runtime/src/kernels/quantize-rows.ts`（設計 =
`docs/research/2026-08-03-dp4a-w8a8-design.md` §4.2）で、このモジュールはその**数値仕様の
鏡像**。重み側（`quantize.fake_quant_int8`）と違って活性は実行時に決まるので、エクスポータで
先に丸めることができない — 参照フィクスチャを採る側で `forward_pre_hook` を掛けて
「ランタイムが実際に食う値」を再現する（ADR 0006 の fake-quant 方法論を活性へ広げた形）。

    s = clamp(rowmax(|x|) / 127, f32 tiny)      … 行 = 最終軸（per-token）
    q = clamp(round(x / s), -127, +127)         … **±127 に閉じる**（−128 不使用）
    x̂ = q · s

MUST: `torch.round` は偶数丸めで、WGSL の `round` と一致する（実測で確認済み）。half-up の
実装（JS の `Math.round` 等）に差し替えると格子の境界で ±1 段ずれる。

MUST: scale はランタイムと同じ **`amax * float32(1/127)` の乗算**で作る（quantize-rows.ts の
MUST の鏡像 — torch の除算 `amax / 127` は正しい丸めなので、乗算形と `s` が 1 ULP 違う行が
数 % 出る。2026-08-31 に乗算形へ揃えた）。残る非鏡像は `x / s` の除算だけで、これは GPU 側が
WGSL の 2.5 ULP 許容を受けるため半整数境界の近傍で ±1 段揺れうる（quantize-rows.ts §除算の
精度 — ビット一致はそこでだけ成立しない）。
"""

from __future__ import annotations

import torch
from torch import nn

from .quantize import INT8_MAX, QuantizeError

#: i8 ペイロードの 4 詰め（平坦添字）に由来する `k` の整列条件。
PACK_ALIGN = 4


def quantize_rows_parts(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """{@link quantize_rows} を `(整数値, 行 scale)` の 2 つに分けて返す。

    整数縮約（`(Σ q_a·q_w)·(s_a·s_w)` 形）をシミュレートする側は `q·s` へ畳む前の整数が要る。
    fake-quant と**同じ 1 本の式**から取り出すことで、両者が別々に仕様から外れる余地を消す。
    """
    amax = x.abs().amax(dim=-1, keepdim=True)
    # MUST: 除算でなく f32(1/127) との乗算（ランタイム鏡像 — モジュール docstring）。
    # python float は weak scalar として amax の dtype（f32）へ落ちてから乗算される。
    scale = torch.clamp(amax * (1.0 / INT8_MAX), min=torch.finfo(torch.float32).tiny)
    return torch.round(x / scale).clamp_(-INT8_MAX, INT8_MAX), scale


def quantize_rows(x: torch.Tensor) -> torch.Tensor:
    """最終軸を行とみなした per-token symmetric i8 の fake-quant。"""
    quantized, scale = quantize_rows_parts(x)
    return quantized * scale


def is_eligible(module: nn.Module) -> bool:
    """ランタイムの適格条件の鏡像（`nn.Linear` で `k % 4 == 0`・bias の有無は問わない）。

    ランタイム側はこれに加えて「重みが i8 で GPU 常駐」を要求する（`linearCompute: "a8"`）。
    i8 系列では適格な `nn.Linear` の重みは全て i8 常駐になるので、本数はここの計数と一致する。
    """
    return isinstance(module, nn.Linear) and module.in_features % PACK_ALIGN == 0


def attach_act_quant(model: nn.Module) -> tuple[list[object], int]:
    """適格 `nn.Linear` の入力を per-token i8 へ落とすフックを掛ける。

    戻り値は `(解除ハンドル, 掛けた本数)`。**本数を返すのが要点**で、0 本のまま参照を採ると
    「w8a8 のつもりで w8 の数を採った」ことに気づけない（ADR 0006 の診断常設と同じ流儀）。
    """
    handles: list[object] = []
    attached = 0

    def make_pre(name: str):
        def pre(_module: nn.Module, args: tuple[object, ...]):
            # MUST: 丸められない呼び出しは素通しでなく例外（姉妹の
            # `quant_calib._make_pre_hook` と同じ規律）。素通しするとその層だけ活性量子化が
            # 外れた参照が「掛けた本数」の診断を通り抜けて w8 のまま採られ、tolerance 門は
            # その差を量子化誤差と区別できない。
            if not args or not isinstance(args[0], torch.Tensor):
                raise QuantizeError(
                    f"'{name}': 活性量子化の対象 linear が位置引数の Tensor で呼ばれていない"
                    "（入力を丸められないので、この層だけ w8 の参照になる）"
                )
            return (quantize_rows(args[0]), *args[1:])

        return pre

    for name, module in model.named_modules():
        if not is_eligible(module):
            continue
        handles.append(module.register_forward_pre_hook(make_pre(name), with_kwargs=False))
        attached += 1
    return handles, attached


def detach_act_quant(handles: list[object]) -> None:
    """{@link attach_act_quant} が返したハンドルを全て外す。"""
    for handle in handles:
        remove = getattr(handle, "remove", None)
        if remove is not None:
            remove()
    handles.clear()
