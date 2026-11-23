"""Script that writes the fixture the host's audio front end is held to: the feature extractor's
settings as the pinned snapshot states them, and the SHA-256 of the frames upstream's arithmetic
gives the multimodal fixture's waveforms with the FFT in f64.

    uv run --with 'transformers==5.23.0' python -m embeddinggemma2.audio_front_end

## What it records

- `featureExtractor` — every setting of the pinned `preprocessor_config.json` that changes a frame
  (the keys of `distribution.EMBEDDINGGEMMA2_AUDIO_FEATURE_EXTRACTOR`: the class, the ones the host
  implements and the ones it requires off), with the values the file states.
- `maxSamples` — the longest audio the extractor reads: its `max_length` call default, not a
  setting of the file (read from the signature).
- `frameDigests` — per waveform of {@link FIXTURE_WAVEFORMS}, the SHA-256 of the `[4n, 128]` f32
  frames (the `L` valid ones of {@link upstream_arithmetic_frames} with the FFT in f64, then
  `4n − L` zero frames — the tower's input as the host builds it).

`packages/models/tests/embeddinggemma2_audio_input_test.ts` holds the host's constants
(`AUDIO_FEATURE_EXTRACTOR`, `AUDIO_MAX_SAMPLES`) and its frames to this file, and
`tests/test_audio_front_end.py` holds the recipe's (`export_audio.EXTRACTOR_SETTINGS`,
`export_audio.MAX_SAMPLES`, the distribution's mirror) and recomputes the digests; neither needs
the snapshot.

## Upstream's arithmetic

`Gemma4AudioFeatureExtractor._extract_spectrogram` on f32 samples (numpy 2): each frame times the
window is an f32 array times an f32 array, so every product is rounded to f32; `np.fft.rfft` of
that f32 array runs in complex64; then the magnitude, the f64 mel product, `log(mel + mel_floor)`
and a cast to f32. {@link upstream_arithmetic_frames} spells these steps with the extractor's own
window and mel filters. With the FFT in f32 it is the extractor's run bit for bit (checked on every
waveform before anything is written); with the FFT in f64 it is the host's arithmetic — the same
f32 products (a correctly rounded f32 multiply has one result, so they are upstream's exactly) and
an FFT that adds no rounding of its own at f32's scale.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from _shared.paths import REPO_ROOT
from embeddinggemma2 import distribution, export_audio, sources
from embeddinggemma2.multimodal_reference import FIXTURE_WAVEFORMS, fixture_json
from embeddinggemma2.waveforms import WAVEFORMS, soft_tokens, valid_frames

#: The fixture (tracked in git).
FIXTURE = (
    REPO_ROOT
    / "packages"
    / "models"
    / "tests"
    / "fixtures"
    / "embeddinggemma2"
    / "audio-front-end.json"
)

#: The snapshot file the settings come from (pinned).
PREPROCESSOR_FILE = "preprocessor_config.json"


class AudioFrontEndError(ValueError):
    """The extractor or the arithmetic is not what the fixture records (fails before writing)."""


def extractor_settings(preprocessor: Mapping[str, Any]) -> dict[str, Any]:
    """The settings the fixture records, as `preprocessor` states them (a missing one fails)."""
    missing = [
        key
        for key in distribution.EMBEDDINGGEMMA2_AUDIO_FEATURE_EXTRACTOR
        if key not in preprocessor
    ]
    if missing:
        raise AudioFrontEndError(f"{PREPROCESSOR_FILE} lacks {missing}")
    return {key: preprocessor[key] for key in distribution.EMBEDDINGGEMMA2_AUDIO_FEATURE_EXTRACTOR}


def extractor_of(settings: Mapping[str, Any]) -> Any:
    """Upstream's feature extractor built from `settings` (`from_dict`, as `from_pretrained` builds
    it from the file), each setting checked on the built object."""
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor

    if settings.get("feature_extractor_type") != Gemma4AudioFeatureExtractor.__name__:
        raise AudioFrontEndError(
            f"the feature extractor is {settings.get('feature_extractor_type')!r}"
        )
    extractor = Gemma4AudioFeatureExtractor.from_dict(dict(settings))
    for key, expected in settings.items():
        if key == "feature_extractor_type":
            continue
        found = getattr(extractor, key)
        found = found.item() if isinstance(found, np.ndarray) else found
        if found != expected:
            raise AudioFrontEndError(f"the built extractor's {key} is {found!r}, not {expected!r}")
    return extractor


def max_samples_of(extractor: Any) -> int:
    """The extractor's `max_length` call default (the longest audio it reads)."""
    return int(inspect.signature(type(extractor).__call__).parameters["max_length"].default)


def upstream_arithmetic_frames(
    extractor: Any, samples: np.ndarray, fft: type[np.floating]
) -> np.ndarray:
    """The `[L, 128]` f32 valid frames of `samples` (f32, 16 kHz mono) by upstream's arithmetic
    (module doc), the FFT in `fft` (`np.float32`: upstream's own; `np.float64`: the host's).

    MUST: the window is f32 — the product's rounding is the one upstream makes only then.
    """
    if samples.dtype != np.float32 or extractor.window.dtype != np.float32:
        raise AudioFrontEndError(
            f"samples {samples.dtype}, window {extractor.window.dtype}: not f32"
        )
    frame_length = int(extractor.frame_length)
    hop = int(extractor.hop_length)
    padded = np.concatenate([np.zeros(frame_length // 2, np.float32), samples])
    frames = np.stack(
        [
            padded[hop * index : hop * index + frame_length]
            for index in range(valid_frames(samples.shape[0]))
        ]
    )
    windowed = (frames * extractor.window).astype(fft)
    magnitude = np.abs(np.fft.rfft(windowed, n=int(extractor.fft_length), axis=-1))
    return np.log(magnitude @ extractor.mel_filters + extractor.mel_floor).astype(np.float32)


def tower_input_digest(frames: np.ndarray) -> str:
    """The SHA-256 of `[4n, 128]` little-endian f32: the `L` frames, then `4n − L` zero frames."""
    count = int(frames.shape[0])
    padded = np.zeros((4 * soft_tokens(count), int(frames.shape[1])), dtype="<f4")
    padded[:count] = frames
    return hashlib.sha256(padded.tobytes()).hexdigest()


def assert_reproduces_upstream(extractor: Any, name: str, samples: np.ndarray) -> None:
    """With the FFT in f32 the arithmetic is the extractor's own run, bit for bit."""
    output = extractor([samples], return_tensors="np")
    count = int(output["input_features_mask"][0].sum())
    upstream = output["input_features"][0][:count]
    ours = upstream_arithmetic_frames(extractor, samples, np.float32)
    if ours.shape != upstream.shape or not np.array_equal(ours, upstream):
        raise AudioFrontEndError(
            f"{name}: the arithmetic with an f32 FFT is not the extractor's run"
        )


def frame_digests(extractor: Any) -> dict[str, str]:
    """{@link tower_input_digest} of the host's arithmetic per fixture waveform, after
    {@link assert_reproduces_upstream} on each."""
    digests: dict[str, str] = {}
    for name in FIXTURE_WAVEFORMS:
        samples = WAVEFORMS[name].f32()
        assert_reproduces_upstream(extractor, name, samples)
        digests[name] = tower_input_digest(
            upstream_arithmetic_frames(extractor, samples, np.float64)
        )
    return digests


def fixture_document(directory: Path) -> dict[str, Any]:
    """The fixture from the pinned snapshot in `directory` (module doc)."""
    import transformers

    export_audio.assert_transformers_version(transformers.__version__)
    sources.assert_pinned(directory, (PREPROCESSOR_FILE,))
    preprocessor = json.loads((directory / PREPROCESSOR_FILE).read_text(encoding="utf-8"))
    settings = extractor_settings(preprocessor)
    extractor = extractor_of(settings)
    return {
        "_doc": [
            "The audio front end of EmbeddingGemma 2 as the TypeScript host is held to it"
            " (generated by tools/export-recipes/embeddinggemma2/audio_front_end.py).",
            "`featureExtractor`: the settings of the pinned preprocessor_config.json that change a"
            " frame, as the file states them. `maxSamples`: the extractor's max_length call"
            " default (not a setting of the file).",
            "`frameDigests`: per waveform of multimodal-reference.safetensors, the SHA-256 of the"
            " [4n, 128] little-endian f32 tower input (the L valid frames, then zero frames to 4n)"
            " by upstream's arithmetic with the FFT in f64: the f32 product of the samples and"
            " the f32 window (upstream's own), numpy's rfft in f64, the magnitude, the f64 mel"
            " product, log(mel + mel_floor), cast to f32. With the FFT in f32 the same steps"
            " reproduce the extractor's frames bit for bit (checked when written).",
        ],
        "source": {
            "upstream": sources.PIN.label,
            "file": PREPROCESSOR_FILE,
            "sha256": sources.PIN.sha256[PREPROCESSOR_FILE],
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "featureExtractor": settings,
        "maxSamples": max_samples_of(extractor),
        "frameDigests": frame_digests(extractor),
    }


def main() -> None:
    document = fixture_document(sources.SNAPSHOT_DIR)
    FIXTURE.write_text(fixture_json(document), encoding="utf-8")
    print(json.dumps({"fixture": str(FIXTURE), "frameDigests": document["frameDigests"]}, indent=1))


if __name__ == "__main__":
    main()
