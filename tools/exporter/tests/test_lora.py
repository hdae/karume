"""LoRA 焼き込みの振る舞い固定（ADR 0016）。

主張は 2 つ: ①ΔW=(B@A)·scale が f32 で計算されて元の重みへ加算されること ②解決できない
対象・形の食い違いは**必ず例外**になること（黙って読み飛ばすと「一部の層だけ LoRA が乗った
劣化モデル」が何事もなく出てくる）。

`load_lora_state_dict` は diffusers 同梱の命名変換を呼ぶので、ここでは扱わない
（変換表は上流が正 — 自前で持たない）。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from karume.lora import fuse_lora


class Target(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 3, bias=False) for _ in range(2)])


def _state(scale_b: float = 1.0, *, rank: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    state: dict[str, torch.Tensor] = {}
    for index in range(2):
        head = f"transformer.blocks.{index}"
        state[f"{head}.lora_A.weight"] = torch.randn(rank, 4, generator=generator)
        state[f"{head}.lora_B.weight"] = torch.randn(3, rank, generator=generator) * scale_b
    return state


class TestFuseLora:
    def test_the_delta_is_added_to_every_target_weight(self):
        model = Target()
        before = [block.weight.detach().clone() for block in model.blocks]
        state = _state()

        report = fuse_lora(model, state, "transformer", scale=0.5)

        assert report.merged == 2
        for index, block in enumerate(model.blocks):
            delta = (
                state[f"transformer.blocks.{index}.lora_B.weight"]
                @ state[f"transformer.blocks.{index}.lora_A.weight"]
            ) * 0.5
            torch.testing.assert_close(block.weight, before[index] + delta)

    def test_the_scale_actually_scales(self):
        """倍率が効いていない実装は scale を変えても同じ重みになる（恒真化の検出）。"""
        half, full = Target(), Target()
        full.load_state_dict(half.state_dict())
        state = _state()

        fuse_lora(half, state, "transformer", scale=0.5)
        fuse_lora(full, state, "transformer", scale=1.0)

        assert not torch.equal(half.blocks[0].weight, full.blocks[0].weight)

    def test_an_all_zero_lora_b_is_reported_as_a_noop(self):
        """成分ごとの ΔW=0 は正常でありうる — 例外にはせず戻り値で知らせる。"""
        model = Target()

        report = fuse_lora(model, _state(scale_b=0.0), "transformer")

        assert report.is_noop
        assert "学習されていない" in report.describe()

    def test_a_prefix_with_no_targets_is_rejected(self):
        with pytest.raises(ValueError, match="1 件も無い"):
            fuse_lora(Target(), _state(), "text_conditioner")

    def test_a_missing_lora_b_is_rejected(self):
        state = _state()
        del state["transformer.blocks.1.lora_B.weight"]

        with pytest.raises(ValueError, match="揃っていない"):
            fuse_lora(Target(), state, "transformer")

    def test_a_target_absent_from_the_model_is_rejected(self):
        """命名変換の取りこぼしは「一部だけ乗った劣化モデル」になるので必ず落とす。"""
        state = _state()
        for suffix in ("lora_A", "lora_B"):
            key = f"transformer.blocks.1.{suffix}.weight"
            state[key.replace("blocks.1", "blocks.9")] = state.pop(key)

        with pytest.raises(ValueError, match="モデルに無い"):
            fuse_lora(Target(), state, "transformer")

    def test_a_leftover_key_under_the_prefix_is_rejected(self):
        """A/B ペアとして消費されないキーは「焼き込まれない成分」— 黙って捨てない。

        `lora_A` を起点に走査するだけの実装ではここが素通りする（`.alpha` や別サフィックスの
        重みが配布形式に混ざっても、宣言どおり「解決できない対象は必ず例外」にならない）。
        """
        state = _state()
        state["transformer.blocks.0.lora_C.weight"] = torch.zeros(3, 4)

        with pytest.raises(ValueError, match="消費できないキー"):
            fuse_lora(Target(), state, "transformer")

    def test_a_shape_mismatch_is_rejected(self):
        state = _state()
        state["transformer.blocks.0.lora_A.weight"] = torch.randn(2, 5)

        with pytest.raises(ValueError, match="形が違う"):
            fuse_lora(Target(), state, "transformer")
