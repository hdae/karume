"""Model card rendering of the vowel-detection distribution (`vowel_detector.card`) — the checks
that exist only on the card side.

The real distribution's `karume.json` is not used — the job here is to see that the fake
manifest's values appear in the body as they are (= that not a single hand-written number is mixed
in), so building it with values **different** from the real ones detects more.

The derivation from the manifest (tables, numbers, usage) is checked **over one assembly pass** by
`TestVowelDetectorModelCard` in `vowel_detector/tests/test_distribution.py`. Until now the
vowel-detection template had no section of its own in the core's `tests/test_modelcard.py`, so no
`test_card.py` existed; but "the advertised load entry point" is a card-only check that can be
observed without a full assembly pass, so it goes here (2026-08-30).
"""

from __future__ import annotations

from typing import Any

from vowel_detector.card import (
    VOWEL_DETECTOR_SUPPORTED_PIPELINE,
    render_vowel_detector_model_card,
)

#: Repository ID spelled in the usage snippet (dist passes it from the pipeline's declaration).
REPO = "hdae/fake-repo"


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


def _container(*refs: dict[str, Any]) -> dict[str, Any]:
    """One dtype of weights (`container` of `karume/5` — ADR 0109 decision 3).

    The card does not read `descriptor`, so only its shape follows the real one (the lengths and
    sha256 of the 2 descriptors).
    """
    return {
        "container": {
            "descriptor": {
                "graph": {"length": 40, "sha256": "0" * 64},
                "model": {"length": 24, "sha256": "1" * 64},
            },
            "parts": list(refs),
        }
    }


def _vowel_detector_manifest() -> dict[str, Any]:
    """Minimal vowel-detection manifest (fake values that do not overlap with the real ones)."""
    return {
        "format": "karume/5",
        "generator": "karume/9.9.9",
        "defaultModel": "ZA",
        "models": {
            "ZA": {
                "pipeline": VOWEL_DETECTOR_SUPPORTED_PIPELINE,
                "weights": {
                    "detector": {
                        "f32": _container(_ref("ZA/detector/model.f32-00001-of-00003.krm", 11, "a"))
                    },
                },
                "assets": {},
                "quants": {"f32": {"weights": {"detector": "f32"}, "session": {}}},
                "defaultQuant": "f32",
                "pipelineConfig": {
                    "sampleRate": 3200,
                    "featureDim": 17,
                    "minFrames": 6,
                    "maxFrames": 1900,
                    "classes": ["a", "i", "u", "e", "o", "N", "pau", "cons"],
                },
            }
        },
    }


class TestVowelDetectorEntryPoint:
    """The load entry point the card advertises — `fromPretrained` only."""

    def test_it_does_not_advertise_the_local_asset_entry_point(self) -> None:
        """`fromAssets` is not advertised (ruling of 2026-08-29).

        The split distribution form can be read as well now (X2-101), but that surface is meant
        for local debugging and assumes the caller holds the bytes; the ordinary entry point for a
        reader using HF is `fromPretrained`. Listing both would leave the reader to decide which
        one to use. `fromPretrained` is checked as well so that the check does not pass when the
        whole Usage section disappears.
        """
        card = render_vowel_detector_model_card(_vowel_detector_manifest(), REPO)
        assert "fromAssets" not in card
        assert "await Pipeline.fromPretrained" in card


class TestVowelDetectorLengthRange:
    """The length range the card states — both bounds come from the manifest's `minFrames` /
    `maxFrames`."""

    @staticmethod
    def _card(min_frames: int) -> str:
        manifest = _vowel_detector_manifest()
        config = manifest["models"]["ZA"]["pipelineConfig"]
        config["sampleRate"] = 16000
        config["minFrames"] = min_frames
        return render_vowel_detector_model_card(manifest, REPO)

    def test_it_states_the_lower_bound_the_pipeline_enforces(self) -> None:
        """The pipeline refuses anything below `minFrames` — a card saying "any length" would
        promise more than it delivers."""
        card = self._card(4)

        assert "any length" not in card
        assert "**One graph for every length from 40 ms to 19.0 s.**" in card
        assert "- **length**: from **4 frames** (40 ms) up to **1900 frames of 10 ms**" in card

    def test_the_lower_bound_follows_the_manifest(self) -> None:
        card = self._card(10)

        assert "from 100 ms to 19.0 s" in card
        assert "from **10 frames** (100 ms)" in card


class TestVowelDetectorFirstParty:
    def test_the_card_carries_no_house_sentence(self) -> None:
        """Its upstream is our own release (`hdae/vowel-detector`): the house non-endorsement
        sentence would disclaim ourselves, so this card alone goes without it."""
        card = render_vowel_detector_model_card(_vowel_detector_manifest(), REPO)
        assert "unofficial conversion" not in card
        assert "endorsed" not in card
