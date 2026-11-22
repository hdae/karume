"""The synthetic waveforms of the audio path (`embeddinggemma2/waveforms.py`).

Pinned here: each waveform is deterministic 16-bit PCM of its stated length, read as `pcm / 32768`
in f32; the module table's frame and soft-token counts are what upstream's feature extractor gives
(its frame mask, at the snapshot's settings); the table covers every residue of the frame count
mod 4 and the front end's 30 s maximum. The cases that need upstream's extractor SKIP without
transformers at the pinned release.
"""

from __future__ import annotations

from importlib import metadata

import numpy as np
import pytest

from embeddinggemma2 import export
from embeddinggemma2.waveforms import PCM_SCALE, WAVEFORMS, soft_tokens, valid_frames

#: The module table: name → (samples, valid frames, soft tokens).
TABLE = {
    "sweep": (16_000, 99, 25),
    "noise": (40_000, 249, 63),
    "click": (16_100, 100, 25),
    "edge": (16_400, 102, 26),
    "tones": (160_000, 999, 250),
    "long": (480_000, 2_999, 750),
}


def _extractor():
    """Upstream's feature extractor at its defaults (the snapshot's `preprocessor_config.json`
    holds the same values — `export_audio.load_feature_extractor` checks them)."""
    pytest.importorskip("transformers")
    if metadata.version("transformers") != export.REQUIRED_TRANSFORMERS:
        pytest.skip(f"transformers {export.REQUIRED_TRANSFORMERS} is not installed")
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor

    return Gemma4AudioFeatureExtractor()


class TestSamples:
    def test_each_waveform_is_pcm16_of_its_length(self) -> None:
        for waveform in WAVEFORMS.values():
            pcm = waveform.pcm16()

            assert pcm.dtype == np.int16
            assert pcm.shape == (waveform.samples,)

    def test_the_f32_samples_are_the_pcm_over_32768_exactly(self) -> None:
        for waveform in WAVEFORMS.values():
            samples = waveform.f32()

            assert samples.dtype == np.float32
            assert np.array_equal(samples.astype(np.float64) * PCM_SCALE, waveform.pcm16())

    def test_each_waveform_is_the_same_bytes_every_time(self) -> None:
        for waveform in WAVEFORMS.values():
            assert waveform.pcm16().tobytes() == waveform.pcm16().tobytes()

    def test_no_waveform_is_silent(self) -> None:
        for waveform in WAVEFORMS.values():
            assert int(np.count_nonzero(waveform.pcm16())) > 0, waveform.name

    def test_edge_ends_at_full_level(self) -> None:
        """Its last real frames carry the signal — the frame mask matters only then."""
        tail = WAVEFORMS["edge"].f32()[-320:]

        assert float(np.abs(tail).max()) > 0.3

    def test_the_names_are_the_tables(self) -> None:
        assert list(WAVEFORMS) == list(TABLE)


class TestCounts:
    def test_each_waveform_has_its_table_row(self) -> None:
        for name, (samples, frames, tokens) in TABLE.items():
            assert WAVEFORMS[name].samples == samples
            assert valid_frames(samples) == frames, name
            assert soft_tokens(frames) == tokens, name

    def test_the_table_covers_every_residue_and_the_maximum(self) -> None:
        assert {frames % 4 for _, frames, _ in TABLE.values()} == {0, 1, 2, 3}
        assert max(tokens for *_, tokens in TABLE.values()) == 750

    @pytest.mark.parametrize(
        ("samples", "frames", "tokens"),
        [(0, 0, 0), (160, 0, 0), (161, 1, 1), (320, 1, 1), (480, 2, 1), (1_000, 6, 2)],
    )
    def test_the_short_lengths(self, samples: int, frames: int, tokens: int) -> None:
        """Upstream's observed counts: 160 samples give no token, 320 one, 1,000 two."""
        assert valid_frames(samples) == frames
        assert soft_tokens(frames) == tokens


class TestUpstreamExtractor:
    def test_each_waveform_has_the_table_s_valid_frames(self) -> None:
        extractor = _extractor()
        for name, (_samples, frames, _tokens) in TABLE.items():
            output = extractor([WAVEFORMS[name].f32()], return_tensors="np")

            mask = output["input_features_mask"][0]
            assert int(mask.sum()) == frames, name
            assert bool(mask[:frames].all()), name

    @pytest.mark.parametrize("samples", [161, 320, 321, 480, 481, 1_000, 16_399, 16_401])
    def test_the_frame_rule_is_the_extractor_s(self, samples: int) -> None:
        extractor = _extractor()
        output = extractor([np.full(samples, 0.25, dtype=np.float32)], return_tensors="np")

        assert int(output["input_features_mask"][0].sum()) == valid_frames(samples)
