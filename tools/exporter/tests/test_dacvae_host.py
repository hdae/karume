"""`dacvae_host.py` の台本レベルの約束事（実重み・音声ライブラリ不要分）。

LUFS 測定と WAV の読み書きは audiotools / soundfile の実経路なのでここでは回さない
（実測は emit のたびに走る常設門が受け持つ）。ここで固定するのは、壊れると**偽 PASS** に
なる側の規律だけ:

- 正規化の**分解**（LUFS 利得 × peak 利得）が波形を再現できなければ落ちる
- ケースが狙った枝を踏んでいなければ落ちる（短尺 / peak > 1 / reflect pad の端数）
- 末尾トリムが「常に長さを返す」実装でも「常に 0 を返す」実装でも落ちる
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

import dacvae_host as host
import export_dacvae as ex


def _cases(**overrides: dict[str, float]) -> dict[str, dict[str, float]]:
    """`_branch_evidence` が読む最小のケース表（既定は全ての枝を正しく踏んでいる形）。"""
    base = {
        "ref-default": {"refDb": -15.7, "peakGain": 1.0, "padSamples": 0, "seconds": 7.6},
        "ref-plain": {"refDb": -14.2, "peakGain": 1.0, "padSamples": 0, "seconds": 3.0},
        "ref-hot": {"refDb": -4.7, "peakGain": 0.33, "padSamples": 0, "seconds": 3.0},
        "ref-short": {"refDb": -68.9, "peakGain": 0.54, "padSamples": 0, "seconds": 0.4},
        "ref-odd": {"refDb": -14.3, "peakGain": 1.0, "padSamples": 1143, "seconds": 3.016},
    }
    for name, values in overrides.items():
        base[name] = {**base[name], **values}
    return base


class TestDecompositionEvidence:
    """MUST: TS 側は audiotools を持たない — 「利得 2 本が正規化の全て」が移植の前提。"""

    def test_a_faithful_decomposition_measures_zero(self):
        raw = torch.linspace(-1.0, 1.0, 16)
        scalars = {"loudnessGain": 0.5, "peakGain": 0.25}

        assert host._decomposition_evidence(raw, raw * 0.5 * 0.25, scalars, "テスト") == 0.0

    def test_a_hidden_extra_step_fails_loudly(self):
        raw = torch.linspace(-1.0, 1.0, 16)
        scalars = {"loudnessGain": 0.5, "peakGain": 0.25}

        with pytest.raises(AssertionError, match="利得 2 本で正規化後の波形を再現できない"):
            host._decomposition_evidence(raw, raw * 0.5 * 0.25 + 1e-4, scalars, "テスト")


class TestBranchEvidence:
    """MUST: ケースを増やしても「どれも既定経路だった」で緑にならないようにする。"""

    def test_the_default_table_passes(self):
        evidence = host._branch_evidence(_cases(), 1920)

        assert evidence == {"shortSeconds": 0.4, "hotPeakGain": 0.33, "oddPadSamples": 1143}

    def test_a_short_case_over_half_a_second_fails_loudly(self):
        with pytest.raises(AssertionError, match="ゼロ pad 測定枝"):
            host._branch_evidence(_cases(**{"ref-short": {"seconds": 0.7}}), 1920)

    def test_a_short_case_with_the_same_loudness_fails_loudly(self):
        with pytest.raises(AssertionError, match="測定が長さに依っていない"):
            host._branch_evidence(_cases(**{"ref-short": {"refDb": -15.7}}), 1920)

    def test_a_hot_case_that_is_not_scaled_fails_loudly(self):
        with pytest.raises(AssertionError, match="peak > 1 の枝"):
            host._branch_evidence(_cases(**{"ref-hot": {"peakGain": 1.0}}), 1920)

    def test_a_plain_case_that_is_scaled_fails_loudly(self):
        with pytest.raises(AssertionError, match="peak ≤ 1 で縮んでいる"):
            host._branch_evidence(_cases(**{"ref-plain": {"peakGain": 0.9}}), 1920)

    def test_a_padded_default_case_fails_loudly(self):
        with pytest.raises(AssertionError, match="hop 1920 の倍数でなくなっている"):
            host._branch_evidence(_cases(**{"ref-default": {"padSamples": 7}}), 1920)

    def test_an_odd_case_without_padding_fails_loudly(self):
        with pytest.raises(AssertionError, match="端数の枝"):
            host._branch_evidence(_cases(**{"ref-odd": {"padSamples": 0}}), 1920)


def _always(value: int):
    """`find_flattening_point` の代役（常に同じ値を返す故障注入）。"""
    return lambda latent: value


def _with_silent_tail(frames: int, tail: int) -> torch.Tensor:
    latent = torch.ones(1, frames, 2)
    if tail:
        latent[:, -tail:] = 0.0
    return latent


class TestTrimEvidence:
    def test_the_upstream_heuristic_is_reported(self):
        latents = {
            "z-full": _with_silent_tail(161, 0),
            "z-no-ref": _with_silent_tail(116, 0),
            "z-silent-tail": _with_silent_tail(161, 40),
        }

        points = host._trim_evidence(_trailing_zero_point, latents)

        assert {name: value["point"] for name, value in points.items()} == {
            "z-full": 161,
            "z-no-ref": 116,
            "z-silent-tail": 121,
        }

    def test_an_implementation_that_never_trims_fails_loudly(self):
        latents = {name: torch.zeros(1, 161, 2) for name, _s, _t in host.TRIM_CASES}

        with pytest.raises(AssertionError, match="末尾トリムの前提"):
            host._trim_evidence(_always(161), latents)

    def test_an_implementation_that_always_trims_fails_loudly(self):
        latents = {name: torch.zeros(1, 161, 2) for name, _s, _t in host.TRIM_CASES}

        with pytest.raises(AssertionError, match="末尾トリムの前提"):
            host._trim_evidence(_always(0), latents)


def _trailing_zero_point(latent: torch.Tensor) -> int:
    """代役の `find_flattening_point`: 末尾の 0 フレーム数を数える（実物と同じ向きの応答）。"""
    zeros = 0
    for index in range(latent.shape[0] - 1, -1, -1):
        if float(latent[index].abs().max()) != 0.0:
            break
        zeros += 1
    return latent.shape[0] - zeros


class TestTrimLatents:
    @pytest.fixture
    def latent_dir(self, tmp_path):
        torch.manual_seed(0)
        for name, frames in (("full", 161), ("no-ref", 116)):
            save_file(
                {ex.LATENT_KEY: torch.randn(1, frames, 2)},
                str(tmp_path / f"{ex.LATENT_CASE_PREFIX}{name}{host.CASE_SUFFIX}"),
            )
        return tmp_path

    def test_the_synthetic_case_zeroes_only_the_tail(self, latent_dir):
        latents = host._trim_latents(latent_dir)

        source = latents["z-full"]
        silent = latents["z-silent-tail"]
        assert torch.equal(silent[:, :-40], source[:, :-40])
        assert float(silent[:, -40:].abs().max()) == 0.0
        # 供給元は書き換えない（同じ実 z を 2 ケースが共有する）。
        assert float(source[:, -40:].abs().max()) > 0.0

    def test_a_missing_latent_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="実 latent が無い"):
            host._trim_latents(tmp_path)


class TestSliceReference:
    def test_seconds_and_extra_samples_are_taken_from_the_head(self):
        wav = torch.arange(10.0)

        sliced = host._slice_reference(wav, 2, 3.0, 1, 1.0)

        assert torch.equal(sliced, wav[:7])

    def test_the_amplitude_scale_is_applied(self):
        wav = torch.ones(10)

        assert float(host._slice_reference(wav, 2, None, 0, 3.0).max()) == 3.0

    def test_a_slice_longer_than_the_source_fails_loudly(self):
        with pytest.raises(SystemExit, match="収まらない"):
            host._slice_reference(torch.ones(10), 2, 9.0, 0, 1.0)
