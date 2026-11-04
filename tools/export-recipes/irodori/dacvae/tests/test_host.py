"""Script-level contracts of `irodori/dacvae/host.py` (the part that needs no real weights or
audio libraries).

LUFS measurement and WAV reading / writing are the real paths of audiotools / soundfile, so they
are not run here (the standing checks that run on every emit measure them). What is pinned here
is only the discipline whose breakage would produce a **false PASS**:

- fails if the **decomposition** of the normalization (LUFS gain × peak gain) cannot reproduce the
  waveform
- fails if a case does not take the branch it targets (short / peak > 1 / reflect pad remainder)
- fails for a tail trim that "always returns the length" as well as one that "always returns 0"
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
from safetensors.torch import save_file

from irodori.dacvae import export as ex
from irodori.dacvae import host


def _cases(**overrides: dict[str, float]) -> dict[str, dict[str, float]]:
    """The minimal case table `_branch_evidence` reads (by default every branch is taken
    correctly)."""
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
    """MUST: the TS side has no audiotools — "the two gains are the whole normalization" is the
    premise of the port."""

    def test_a_faithful_decomposition_measures_zero(self):
        raw = torch.linspace(-1.0, 1.0, 16)
        scalars = {"loudnessGain": 0.5, "peakGain": 0.25}

        assert host._decomposition_evidence(raw, raw * 0.5 * 0.25, scalars, "test") == 0.0

    def test_a_hidden_extra_step_fails_loudly(self):
        raw = torch.linspace(-1.0, 1.0, 16)
        scalars = {"loudnessGain": 0.5, "peakGain": 0.25}

        with pytest.raises(
            AssertionError, match="the two gains do not reproduce the normalized waveform"
        ):
            host._decomposition_evidence(raw, raw * 0.5 * 0.25 + 1e-4, scalars, "test")


class TestBranchEvidence:
    """MUST: adding cases must not let "every one was the default path" go green."""

    def test_the_default_table_passes(self):
        evidence = host._branch_evidence(_cases(), 1920)

        assert evidence == {"shortSeconds": 0.4, "hotPeakGain": 0.33, "oddPadSamples": 1143}

    def test_a_short_case_over_half_a_second_fails_loudly(self):
        with pytest.raises(AssertionError, match="zero-pad measurement branch"):
            host._branch_evidence(_cases(**{"ref-short": {"seconds": 0.7}}), 1920)

    def test_a_short_case_with_the_same_loudness_fails_loudly(self):
        with pytest.raises(AssertionError, match="does not depend on the length"):
            host._branch_evidence(_cases(**{"ref-short": {"refDb": -15.7}}), 1920)

    def test_a_hot_case_that_is_not_scaled_fails_loudly(self):
        with pytest.raises(AssertionError, match="the peak > 1 branch"):
            host._branch_evidence(_cases(**{"ref-hot": {"peakGain": 1.0}}), 1920)

    def test_a_plain_case_that_is_scaled_fails_loudly(self):
        with pytest.raises(AssertionError, match="shrinks at peak ≤ 1"):
            host._branch_evidence(_cases(**{"ref-plain": {"peakGain": 0.9}}), 1920)

    def test_a_padded_default_case_fails_loudly(self):
        with pytest.raises(AssertionError, match="no longer a multiple of hop 1920"):
            host._branch_evidence(_cases(**{"ref-default": {"padSamples": 7}}), 1920)

    def test_an_odd_case_without_padding_fails_loudly(self):
        with pytest.raises(AssertionError, match="remainder branch"):
            host._branch_evidence(_cases(**{"ref-odd": {"padSamples": 0}}), 1920)


def _always(value: int):
    """Stand-in for `find_flattening_point` (fault injection that always returns the same
    value)."""
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

        with pytest.raises(AssertionError, match="premise of the tail trim"):
            host._trim_evidence(_always(161), latents)

    def test_an_implementation_that_always_trims_fails_loudly(self):
        latents = {name: torch.zeros(1, 161, 2) for name, _s, _t in host.TRIM_CASES}

        with pytest.raises(AssertionError, match="premise of the tail trim"):
            host._trim_evidence(_always(0), latents)


def _trailing_zero_point(latent: torch.Tensor) -> int:
    """Stand-in `find_flattening_point`: counts the trailing zero frames (responds in the same
    direction as the real one)."""
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
        # The source is not modified (two cases share the same real z).
        assert float(source[:, -40:].abs().max()) > 0.0

    def test_a_missing_latent_fails_loudly(self, tmp_path):
        with pytest.raises(SystemExit, match="real latent missing"):
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
        with pytest.raises(SystemExit, match="does not fit"):
            host._slice_reference(torch.ones(10), 2, 9.0, 0, 1.0)


class TestReadWav:
    """The int16 → f32 convention (/32768) and downmix to mono (channel average) — both are
    premises of the host port."""

    @staticmethod
    def _write(path, values, *, sample_rate=48000):
        soundfile = pytest.importorskip("soundfile")
        import numpy as np

        soundfile.write(str(path), np.array(values, dtype=np.int16), sample_rate, subtype="PCM_16")
        return path

    def test_int16_is_normalized_by_32768(self, tmp_path):
        """The low end −32768 mapping to exactly −1.0 is the /32768 side (/32767 would overflow)."""
        path = self._write(tmp_path / "mono.wav", [-32768, 0, 32767])

        wav, sample_rate = ex.read_wav(path)

        assert sample_rate == 48000
        assert torch.equal(wav, torch.tensor([-1.0, 0.0, 32767 / 32768]))

    def test_a_stereo_file_is_averaged_into_one_channel(self, tmp_path):
        """The same downmix to mono as `encode_waveform` in `codec.py` (average of left and right;
        the channel axis disappears)."""
        path = self._write(tmp_path / "stereo.wav", [[16384, -16384], [8192, 16384], [-32768, 0]])

        wav, _sample_rate = ex.read_wav(path)

        assert wav.shape == (3,)
        assert torch.equal(wav, torch.tensor([0.0, 0.375, -0.5]))


class TestWavScaleEvidence:
    """MUST: the convention is measured, not asserted (on real audio it differs from /32767 by
    only 3e-5 relative)."""

    def test_the_real_reader_passes_and_records_the_divisor(self, tmp_path):
        pytest.importorskip("soundfile")

        evidence = host._wav_scale_evidence(48000)

        assert evidence["int16Divisor"] == 32768
        assert evidence["probeInt16"] == list(host.WAV_SCALE_PROBE)

    def test_a_reader_that_divides_by_32767_fails_loudly(self, monkeypatch):
        """Guarding against tautology: measures that the check looks at "does it follow the
        convention", not "could it be read"."""
        pytest.importorskip("soundfile")
        real = ex.read_wav

        def divided_by_32767(path):
            wav, sample_rate = real(path)
            return wav * (32768 / 32767), sample_rate

        monkeypatch.setattr(ex, "read_wav", divided_by_32767)

        with pytest.raises(AssertionError, match="is not /32768"):
            host._wav_scale_evidence(48000)


class _FakeAudioSignal:
    """A stand-in for only the surface of `audiotools.AudioSignal` that `_normalize_decomposed`
    touches.

    The `target_db=None` path **only records** the LUFS measurement and does not use it for the
    gain — what is observed is that the known value the stand-in returns shows up as `refDb` as is,
    while `loudnessGain` stays 1.0.
    """

    GAIN_FACTOR = 0.1

    def __init__(self, data, sample_rate: int) -> None:
        self.data = data
        self.sample_rate = sample_rate

    def loudness(self) -> float:
        return -12.5


@pytest.fixture
def fake_audiotools(monkeypatch):
    module = types.ModuleType("audiotools")
    module.AudioSignal = _FakeAudioSignal
    monkeypatch.setitem(sys.modules, "audiotools", module)
    return module


class TestNormalizeDecomposedWithoutTarget:
    """`target_db=None` = the path of upstream `encode_waveform` with only the peak safety
    scale."""

    def test_a_waveform_inside_unity_is_left_bit_identical(self, fake_audiotools):
        raw = torch.tensor([0.5, -0.25, 1.0])

        decomposed, scalars = host._normalize_decomposed(raw, 48000, None)

        assert scalars["loudnessGain"] == 1.0
        assert scalars["peakGain"] == 1.0
        assert scalars["refDb"] == -12.5, (
            "measured, but it has no effect on the gain on the None path"
        )
        assert torch.equal(decomposed, raw)

    def test_a_hot_waveform_is_scaled_by_one_over_the_peak(self, fake_audiotools):
        raw = torch.tensor([0.5, -2.0, 1.0])

        decomposed, scalars = host._normalize_decomposed(raw, 48000, None)

        assert scalars["peakBeforeScale"] == 2.0
        assert scalars["peakGain"] == 0.5
        assert scalars["loudnessGain"] == 1.0
        assert torch.equal(decomposed, raw * 0.5)
