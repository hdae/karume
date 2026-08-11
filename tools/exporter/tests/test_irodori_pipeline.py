"""`irodori_pipeline.py` の台本レベルの約束事（実重み不要分）。

実重みの emit は手動（`export_irodori.py` と同じ規律）。ここで固定するのは、**ホストが
写している 2 段**（S 決定と Euler の組み立て）の細部で、壊れると golden が静かに別物になる側:

- t スケジュールが上流の式（`(1 - linspace) * 0.999`）で、単調減少であること
- 銀行家丸めが「四捨五入」に化けていないこと（0.5 の扱い）
- S の clamp が下限 13 / 上限 750 で効くこと
- 区間マスクの割り方（self / text / speaker / caption の順とオフセット）と、uncond が
  **その区間だけ**を落とすこと
- CFG のスケール表が uncond 変種の綴りと 1 対 1 であること
"""

from __future__ import annotations

import math

import pytest
import torch

import export_irodori as ex
import irodori_pipeline as ip


class TestTSchedule:
    def test_it_starts_at_the_init_scale_and_ends_at_zero(self):
        schedule = ip.t_schedule(40)

        assert schedule.shape == (41,)
        assert float(schedule[0]) == pytest.approx(ip.INIT_SCALE)
        assert float(schedule[-1]) == 0.0

    def test_it_is_strictly_decreasing(self):
        """上流 `sample_euler_rf_cfg` が `t_schedule[:-1] > t_schedule[1:]` を要求する。"""
        schedule = ip.t_schedule(40)

        assert bool(torch.all(schedule[:-1] > schedule[1:]))

    def test_the_closed_form_is_within_one_ulp(self):
        """TS 側は閉形式で作る想定 — 差が 1 ulp 級であることを固定する。"""
        assert ip.closed_form_matches(ip.t_schedule(2), 2) == 0.0
        assert ip.closed_form_matches(ip.t_schedule(40), 40) < 1e-7

    def test_the_cfg_window_covers_the_first_half_of_the_steps(self):
        """CFG は `t ∈ [0.5, 1.0]` の step だけ — 40 step では前半 20 本。"""
        schedule = ip.t_schedule(40)
        inside = [
            index for index in range(40) if ip.CFG_MIN_T <= float(schedule[index]) <= ip.CFG_MAX_T
        ]

        assert inside == list(range(20))


class TestBankerRounding:
    """MUST: 「四捨五入」に化けさせない（TS の `Math.round` は 0.5 で常に上へ行く）。"""

    @pytest.mark.parametrize(
        ("frames", "want"),
        [(12.4, 12), (12.5, 12), (12.6, 13), (13.5, 14), (14.5, 14), (160.9389, 161)],
    )
    def test_half_way_values_round_to_even(self, frames, want):
        assert ip.banker_round_frames(frames) == want


class TestSequenceLength:
    def test_a_typical_prediction_is_used_as_is(self):
        # expm1 の逆（log1p）を入れて狙った frames を作る。
        log_frames = torch.log1p(torch.tensor([160.9389]))

        steps, meta = ip.sequence_length(log_frames, ex.CODEC_FRAME_RATE)

        assert steps == 161
        assert meta["minFrames"] == math.ceil(ip.MIN_SECONDS * ex.CODEC_FRAME_RATE)
        assert meta["maxFrames"] == math.floor(ip.MAX_SECONDS * ex.CODEC_FRAME_RATE)

    def test_a_short_prediction_is_clamped_up(self):
        steps, meta = ip.sequence_length(torch.log1p(torch.tensor([1.0])), ex.CODEC_FRAME_RATE)

        assert steps == meta["minFrames"] == 13

    def test_a_long_prediction_is_clamped_down(self):
        steps, meta = ip.sequence_length(torch.log1p(torch.tensor([5000.0])), ex.CODEC_FRAME_RATE)

        assert steps == meta["maxFrames"] == 750

    def test_the_mean_is_taken_over_the_batch(self):
        """上流は `expm1(...).float().mean()`（候補数 > 1 のときの平均）。"""
        log_frames = torch.log1p(torch.tensor([100.0, 200.0]))

        steps, _meta = ip.sequence_length(log_frames, ex.CODEC_FRAME_RATE)

        assert steps == 150


CAPS = {"text": 4, "speaker": 5, "caption": 6}
USED = {"text": 2, "speaker": 3, "caption": 4}


class TestSegmentMasks:
    """MUST: 区間の割り方（順序とオフセット）を固定する — ずれても shape は合ってしまう。"""

    def test_the_cond_mask_marks_every_used_prefix(self):
        mask = ip._segment_masks(3, USED, CAPS, None)

        assert tuple(mask.shape) == (1, 1, 1, 3 + 4 + 5 + 6)
        flat = mask[0, 0, 0].tolist()
        assert flat[:3] == [True] * 3
        assert flat[3:7] == [True, True, False, False]
        assert flat[7:12] == [True, True, True, False, False]
        assert flat[12:] == [True, True, True, True, False, False]

    @pytest.mark.parametrize("uncond", ["text", "speaker", "caption"])
    def test_an_uncond_variant_clears_only_its_own_segment(self, uncond):
        cond = ip._segment_masks(3, USED, CAPS, None)[0, 0, 0]
        got = ip._segment_masks(3, USED, CAPS, uncond)[0, 0, 0]

        offsets = {"text": 3, "speaker": 7, "caption": 12}
        start = offsets[uncond]
        end = start + CAPS[uncond]
        assert not bool(got[start:end].any())
        assert torch.equal(got[:start], cond[:start])
        assert torch.equal(got[end:], cond[end:])

    def test_the_variant_names_match_the_export_script(self):
        """綴りが割れると uncond の区間が黙って別の条件を落とす。"""
        assert tuple(ip.CFG_SCALES) == ex.DIT_UNCOND_VARIANTS


class TestRightPadIds:
    def test_it_pads_to_the_declared_length_and_marks_the_head(self):
        ids = torch.tensor([[1, 5, 6]], dtype=torch.int64)

        padded, mask = ip._right_pad_ids(ids, 6, 3)

        assert padded[0].tolist() == [1, 5, 6, 3, 3, 3]
        assert mask[0].tolist() == [True, True, True, False, False, False]

    def test_a_missing_sequence_becomes_all_pad_with_an_empty_mask(self):
        """caption 空の経路（上流 `caption_mask.zero_()`）。"""
        padded, mask = ip._right_pad_ids(None, 4, 3)

        assert padded[0].tolist() == [3, 3, 3, 3]
        assert not bool(mask.any())


class TestCli:
    def test_the_default_out_dir_is_derived_from_the_weight_directory(self, tmp_path):
        out = ip.default_out_dir(tmp_path / "v4-small")

        assert out.name == "pipeline"
        assert out.parent.name == "irodori-v4-small"

    def test_the_case_names_are_unique(self):
        names = [case.name for case in ip.PIPELINE_CASES]

        assert len(names) == len(set(names))

    def test_the_reference_frames_divide_by_the_speaker_patch_size(self):
        """割り切れないと上流 `patch_sequence_with_mask` が端を捨て、保存した入力とずれる。"""
        for case in ip.PIPELINE_CASES:
            if case.reference is not None:
                assert case.reference.frames % 4 == 0
