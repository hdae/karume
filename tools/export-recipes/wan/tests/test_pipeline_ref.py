"""`wan/pipeline_ref.py` のうち**実重みが要らない**部分の約束事。

壊れても例外が出ず**数だけが静かにずれる**側だけを固定する:

- 埋め込みの注入の形（有効長の後ろをゼロで 512 まで埋める — 上流 `_get_t5_prompt_embeds`）
- 潜在の形から導く動画の大きさ（`4(F−1)+1` / `8H` / `8W`）
- MPS が見える機では組まない（RoPE の表が float32 になる — ADR 0118 決定 3）

読み込みと 1 ステップの疎通は手動（`uv run --group wan python -m wan.pipeline_ref --smoke`）。
"""

from __future__ import annotations

import pytest
import torch

from wan import pipeline_ref


class TestPadTextEmbeds:
    def test_valid_rows_stay_and_the_rest_is_zero(self):
        embeds = torch.randn(7, pipeline_ref.TEXT_DIM)

        padded = pipeline_ref.pad_text_embeds(embeds)

        assert padded.shape == (1, pipeline_ref.MAX_SEQUENCE_LENGTH, pipeline_ref.TEXT_DIM)
        assert torch.equal(padded[0, :7], embeds)
        assert torch.count_nonzero(padded[0, 7:]) == 0

    @pytest.mark.parametrize(
        "embeds",
        [
            torch.zeros(0, 4096),
            torch.zeros(513, 4096),
            torch.zeros(4, 4095),
            torch.zeros(1, 4, 4096),
            torch.zeros(4, 4096, dtype=torch.float16),
        ],
    )
    def test_shapes_and_dtypes_outside_the_contract_fail_loudly(self, embeds: torch.Tensor):
        with pytest.raises(ValueError):
            pipeline_ref.pad_text_embeds(embeds)


class TestVideoSize:
    def test_it_inverts_the_vae_compression(self):
        """832×480・33 フレーム ↔ 潜在 `[16, 9, 60, 104]`（ADR 0118 の受理集合）。"""
        latents = torch.zeros(1, 16, 9, 60, 104)

        assert pipeline_ref.video_size(latents) == (33, 480, 832)

    def test_a_batched_latent_fails_loudly(self):
        with pytest.raises(ValueError):
            pipeline_ref.video_size(torch.zeros(2, 16, 3, 16, 16))


class TestMps:
    def test_a_visible_mps_refuses_to_build_the_reference(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

        with pytest.raises(pipeline_ref.MpsVisibleError):
            pipeline_ref.assert_no_mps()

    def test_this_machine_builds_the_float64_tables(self):
        pipeline_ref.assert_no_mps()
