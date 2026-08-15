"""RoPE バッファの降格と、その空振りを落とす門の回帰テスト（`karume.rope`）。

対象はモデル非依存の走査なので合成 nn.Module だけで固定する（上流パッケージ不要 —
ADR 0065 段 2 で Anima の patch 層から回収した）。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from karume import rope


class TestLiftRopeBuffers:
    def test_it_moves_inv_freq_out_of_the_buffers(self) -> None:
        """`inv_freq` をバッファから素の属性へ降格する（定数畳み込みの葉にするため）。"""

        class Rotary(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("inv_freq", torch.arange(4.0), persistent=False)

        root = nn.Sequential(Rotary())

        lifted = rope.lift_rope_buffers(root)

        assert lifted == 1
        assert "inv_freq" not in dict(root.named_buffers())
        assert torch.equal(root[0].inv_freq, torch.arange(4.0))

    def test_it_matches_layer_type_prefixed_names(self) -> None:
        """layer_type 接頭辞付きの名前も降格する（Gemma3 は `<layer_type>_inv_freq` を持つ）。

        接尾一致にした根拠の検出器。完全一致に戻すとここが 0 本になり、そのモデルの
        export では sin / cos が IR に残る。
        """

        class LayeredRotary(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                for name in (
                    "sliding_attention_inv_freq",
                    "full_attention_inv_freq",
                    "sliding_attention_original_inv_freq",
                    "full_attention_original_inv_freq",
                ):
                    self.register_buffer(name, torch.arange(4.0), persistent=False)
                # 接尾辞を持たないバッファは対象外（走査が無関係な定数まで拾わない）。
                self.register_buffer("attention_scale", torch.ones(1), persistent=False)

        root = LayeredRotary()

        lifted = rope.lift_rope_buffers(root)

        assert lifted == 4
        assert list(dict(root.named_buffers())) == ["attention_scale"]

    def test_a_model_without_rope_buffers_is_rejected(self) -> None:
        """走査が空振りする形は落とす（恒真化の門 — 上流の属性名が変わると静かに壊れる）。"""
        with pytest.raises(ValueError, match="1 本も見つからない"):
            rope.assert_rope_lifted(nn.Linear(2, 2), "テスト")
