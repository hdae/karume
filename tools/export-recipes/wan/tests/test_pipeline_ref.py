"""The contract of `wan/pipeline_ref.py`.

Pins only the side where a breakage raises no exception and **only the numbers drift silently**:

- the shape of the embedding injection (zero padding after the valid length up to 512 — upstream
  `_get_t5_prompt_embeds`)
- no build on a machine where MPS is visible (the RoPE tables would become float32 — ADR 0118
  decision 3)
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


class TestMps:
    def test_a_visible_mps_refuses_to_build_the_reference(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

        with pytest.raises(pipeline_ref.MpsVisibleError):
            pipeline_ref.assert_no_mps()

    def test_this_machine_builds_the_float64_tables(self):
        pipeline_ref.assert_no_mps()
