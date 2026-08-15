"""LoRA を **export 前に重みへ焼き込む**（ADR 0016）。

Karume の成果物は torch.export が出す静的グラフなので、LoRA は実行時の機構ではなく
**エクスポート時の重み変換**として扱う。焼き込んでも IR は 1 ノードも変わらず、ランタイム側の
実装は一切要らない（アダプタ切り替えが要るようになったら、そのときに初めて設計をやり直す）。

対象は Anima の蒸留（turbo 等）LoRA。配布形式は非 diffusers 命名なので、**diffusers 同梱の
変換関数**で diffusers 命名へ直してから使う（命名対応表を自前で持たない — 上流が正）。変換後は
`transformer.*` と `text_conditioner.*` の 2 系統に分かれる。

alpha を持たない配布形式なので倍率は 1.0 が既定（`scale` で上書きできる）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

_A_SUFFIX = ".lora_A.weight"
_B_SUFFIX = ".lora_B.weight"


@dataclass(frozen=True)
class FuseReport:
    """焼き込みの結果。呼び出し側がログに出して人が異常に気づけるようにする。"""

    merged: int
    """焼き込んだ Linear の本数。"""

    max_relative_delta: float
    """`‖ΔW‖ / ‖W‖` の最大値。極端に大きければ倍率がおかしい。"""

    @property
    def is_noop(self) -> bool:
        """この成分は重みが 1 つも動かなかった（`lora_B` が全て 0）。"""
        return self.max_relative_delta == 0.0

    def describe(self) -> str:
        if self.is_noop:
            return (
                f"{self.merged} 層は ΔW=0 — LoRA に含まれるが学習されていない"
                "（この成分は再エクスポート不要）"
            )
        return f"{self.merged} 層に焼き込み（‖ΔW‖/‖W‖ の最大 {self.max_relative_delta:.3f}）"


def load_lora_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """LoRA ファイルを読み、diffusers 命名へ変換する。

    MUST: **ファイル全体が無効**（どの `lora_B` も 0）なら例外にする。LoRA は B を 0 で
    初期化するので、学習されていないファイルや取り違えたファイルは「読めるが何も起きない」形に
    なる — 気づかずに base と同じ絵を出し続ける。成分ごとの 0 は正常でありうるので、そちらは
    {@link fuse_lora} の戻り値で呼び出し側が判断する。
    """
    from diffusers.loaders.lora_conversion_utils import (
        _convert_non_diffusers_anima_lora_to_diffusers as convert,
    )
    from safetensors.torch import load_file

    raw = load_file(str(path))
    if not raw:
        raise ValueError(f"{path}: テンソルが 1 件も無い")
    converted = convert(raw)
    live = sum(
        1
        for key, value in converted.items()
        if key.endswith(_B_SUFFIX) and float(value.abs().max()) > 0
    )
    if live == 0:
        raise ValueError(f"{path}: lora_B が全て 0 — 適用しても base と同じ結果になる")
    return converted


def fuse_lora(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    scale: float = 1.0,
) -> FuseReport:
    """`prefix` 配下の LoRA を `model` の重みへ加算する（in-place）。

    ΔW = (B @ A)·scale を **f32 で計算**してから元 dtype へ足す（低精度で B@A を組むと、
    rank の小さい積で桁落ちしたぶんがそのまま重みの誤差になる）。

    MUST: 解決できない対象・形の食い違いは**必ず例外**にする。黙って読み飛ばすと「一部の層
    だけ LoRA が乗った劣化モデル」が何事もなく出力され、絵が微妙に悪いだけの状態が続く —
    蒸留 LoRA では特に気づきにくい。**prefix 配下のキーは 1 本残らず A/B ペアとして消費**
    することもここで見る（`lora_A` を起点に走査するだけだと、`.alpha` や別サフィックスの
    重みが黙って捨てられる = 宣言と実装が食い違う）。
    """
    head = f"{prefix}."
    unconsumed = {key for key in state_dict if key.startswith(head)}
    targets = sorted(
        key.removeprefix(head).removesuffix(_A_SUFFIX)
        for key in unconsumed
        if key.endswith(_A_SUFFIX)
    )
    if not targets:
        raise ValueError(f"LoRA に {prefix} 向けの層が 1 件も無い")

    params = dict(model.named_parameters())
    max_relative = 0.0
    for target in targets:
        down = state_dict.get(f"{head}{target}{_A_SUFFIX}")
        up = state_dict.get(f"{head}{target}{_B_SUFFIX}")
        if down is None or up is None:
            raise ValueError(f"{prefix}: {target} の lora_A / lora_B が揃っていない")
        unconsumed -= {f"{head}{target}{_A_SUFFIX}", f"{head}{target}{_B_SUFFIX}"}
        weight = params.get(f"{target}.weight")
        if weight is None:
            raise ValueError(f"{prefix}: {target}.weight がモデルに無い（命名変換の取りこぼし）")
        delta = (up.to(torch.float32) @ down.to(torch.float32)) * scale
        if delta.shape != weight.shape:
            raise ValueError(
                f"{prefix}: {target} の形が違う"
                f" LoRA={tuple(delta.shape)} 重み={tuple(weight.shape)}"
            )
        norm = float(torch.linalg.vector_norm(weight.detach()))
        if norm > 0:
            max_relative = max(max_relative, float(torch.linalg.vector_norm(delta)) / norm)
        with torch.no_grad():
            weight.add_(delta.to(weight.dtype))

    if unconsumed:
        raise ValueError(
            f"{prefix}: A/B ペアとして消費できないキーが {len(unconsumed)} 件残った"
            f"（{', '.join(sorted(unconsumed)[:5])}）— 焼き込まれない成分がある"
        )
    return FuseReport(merged=len(targets), max_relative_delta=max_relative)
