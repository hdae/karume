"""The contract of the parts of `wan/pipeline_ref.py` that **need no real weights**.

Pins only the side where a breakage raises no exception and **only the numbers drift silently**:

- the shape of the embedding injection (zero padding after the valid length up to 512 — upstream
  `_get_t5_prompt_embeds`)
- the video size derived from the latent shape (`4(F−1)+1` / `8H` / `8W`)
- no build on a machine where MPS is visible (the RoPE tables would become float32 — ADR 0118
  decision 3)

Loading and a 1-step smoke run are manual (`uv run --group wan python -m wan.pipeline_ref --smoke`).
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
        """832×480, 33 frames ↔ latent `[16, 9, 60, 104]` (the accepted set of ADR 0118)."""
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
