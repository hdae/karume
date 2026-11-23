"""The fixture of the host's audio front end (`embeddinggemma2/audio_front_end.py`): the recipe's
settings are the fixture's (no snapshot, no transformers — these never skip), and with transformers
at the pinned release the fixture's digests are upstream's arithmetic with the FFT in f64, whose
FFT-in-f32 twin is the extractor's own run.

The TypeScript side of the same fixture is
`packages/models/tests/embeddinggemma2_audio_input_test.ts`.
"""

from __future__ import annotations

import json
from importlib import metadata
from typing import Any

import numpy as np
import pytest

from embeddinggemma2 import audio_front_end, distribution, export, export_audio, sources
from embeddinggemma2.waveforms import WAVEFORMS


def _fixture() -> dict[str, Any]:
    return json.loads(audio_front_end.FIXTURE.read_text(encoding="utf-8"))


def _require_transformers() -> None:
    pytest.importorskip("transformers")
    version = metadata.version("transformers")
    if version != export.REQUIRED_TRANSFORMERS:
        pytest.skip(f"transformers {version} is installed, not {export.REQUIRED_TRANSFORMERS}")


def _typed(settings: dict[str, Any]) -> dict[str, tuple[bool, Any]]:
    """Each value with whether it is a bool (`True == 1` in Python; the file's `false` is not 0)."""
    return {key: (isinstance(value, bool), value) for key, value in settings.items()}


class TestSettings:
    def test_the_fixture_states_the_recipe_s_extractor_settings(self) -> None:
        expected = {"feature_extractor_type": "Gemma4AudioFeatureExtractor"}
        expected.update(export_audio.EXTRACTOR_SETTINGS)

        assert _typed(_fixture()["featureExtractor"]) == _typed(expected)

    def test_the_fixture_states_the_distribution_s_mirror(self) -> None:
        mirror = dict(distribution.EMBEDDINGGEMMA2_AUDIO_FEATURE_EXTRACTOR)

        assert _typed(_fixture()["featureExtractor"]) == _typed(mirror)
        assert list(_fixture()["featureExtractor"]) == list(mirror)

    def test_the_longest_audio_is_the_recipe_s(self) -> None:
        assert _fixture()["maxSamples"] == export_audio.MAX_SAMPLES
        assert _fixture()["maxSamples"] == distribution.EMBEDDINGGEMMA2_AUDIO_MAX_SAMPLES

    def test_the_source_is_the_pinned_file(self) -> None:
        source = _fixture()["source"]

        assert source["upstream"] == sources.PIN.label
        assert source["sha256"] == sources.PIN.sha256[audio_front_end.PREPROCESSOR_FILE]

    @pytest.mark.skipif(
        not (sources.SNAPSHOT_DIR / audio_front_end.PREPROCESSOR_FILE).is_file(),
        reason="the pinned snapshot is absent",
    )
    def test_the_settings_are_the_snapshot_file_s(self) -> None:
        sources.assert_pinned(sources.SNAPSHOT_DIR, (audio_front_end.PREPROCESSOR_FILE,))
        path = sources.SNAPSHOT_DIR / audio_front_end.PREPROCESSOR_FILE
        preprocessor = json.loads(path.read_text(encoding="utf-8"))

        assert audio_front_end.extractor_settings(preprocessor) == _fixture()["featureExtractor"]


@pytest.fixture
def extractor() -> Any:
    _require_transformers()
    return audio_front_end.extractor_of(_fixture()["featureExtractor"])


class TestArithmetic:
    """Upstream's extractor built from the fixture's settings (transformers at the pinned
    release)."""

    def test_the_longest_audio_is_the_extractor_s_default(self, extractor) -> None:
        assert audio_front_end.max_samples_of(extractor) == _fixture()["maxSamples"]

    @pytest.mark.parametrize("name", list(_fixture()["frameDigests"]))
    def test_with_an_f32_fft_it_is_the_extractor_s_run(self, extractor, name: str) -> None:
        audio_front_end.assert_reproduces_upstream(extractor, name, WAVEFORMS[name].f32())

    @pytest.mark.parametrize("name", list(_fixture()["frameDigests"]))
    def test_the_digests_are_its_frames_with_an_f64_fft(self, extractor, name: str) -> None:
        frames = audio_front_end.upstream_arithmetic_frames(
            extractor, WAVEFORMS[name].f32(), np.float64
        )

        assert audio_front_end.tower_input_digest(frames) == _fixture()["frameDigests"][name]

    def test_the_fft_precision_is_what_tells_the_two_apart(self, extractor) -> None:
        """A positive control: the f32 FFT's frames are not the digest's (so the digest pins the
        f64 FFT, not merely the product)."""
        frames = audio_front_end.upstream_arithmetic_frames(
            extractor, WAVEFORMS["sweep"].f32(), np.float32
        )

        assert audio_front_end.tower_input_digest(frames) != _fixture()["frameDigests"]["sweep"]

    def test_the_fixture_waveforms_are_the_multimodal_fixture_s(self) -> None:
        from embeddinggemma2.multimodal_reference import FIXTURE_WAVEFORMS

        assert list(_fixture()["frameDigests"]) == list(FIXTURE_WAVEFORMS)
