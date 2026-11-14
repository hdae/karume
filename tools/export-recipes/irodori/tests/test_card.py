"""Model card rendering of the Irodori distribution (`irodori.card`) — checks that exist only on
the card side.

The real distribution's `karume.json` is not used — the job here is to see that the values of a
fake manifest appear in the text as they are (= that not a single hand-written number is mixed
in), so building it from values that **differ** from the real ones detects more.

Derivation from the manifest (tables, numbers, usage) is checked by `TestIrodoriModelCard` in
`irodori/tests/test_distribution.py` over **one full build**. Until now the Irodori template had
no section of its own in the core's `tests/test_modelcard.py`, so there was no `test_card.py`,
but "the load entry point the card points to" is a card-only check that can be observed without
a full build, so it lives here (2026-08-30).
"""

from __future__ import annotations

from typing import Any

import pytest

from irodori.card import IRODORI_SUPPORTED_PIPELINE, render_irodori_model_card

#: The repo ID written into the usage snippet (dist passes it from the pipeline declaration).
REPO = "hdae/fake-repo"


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


def _container(*refs: dict[str, Any]) -> dict[str, Any]:
    """One dtype of `weights` (the `container` of `karume/5` — ADR 0109 decision 3).

    The card does not read `descriptor`, so it only has the real shape (the length and sha256 of
    the 2 documents).
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


#: The model name of the fake manifest. Attribution (upstream repo, display name) is now
#: looked up from the model name in `IRODORI_UPSTREAMS` (2026-09-01), so only this uses a real
#: key — every other value stays fake.
MODEL = "v4.1-small"


def _irodori_manifest(model: str = MODEL) -> dict[str, Any]:
    """Minimal Irodori manifest (every value but the model name is fake and differs from the
    real ones)."""
    return {
        "format": "karume/5",
        "generator": "karume/9.9.9",
        "defaultModel": model,
        "models": {
            model: {
                "pipeline": IRODORI_SUPPORTED_PIPELINE,
                "weights": {
                    "backbone": {
                        "f16": _container(_ref("ZA/backbone/model.f16-00001-of-00003.krm", 11, "a"))
                    },
                    "dit": {
                        "f16": _container(_ref("ZA/dit/model.f16-00001-of-00003.krm", 13, "b"))
                    },
                },
                "assets": {},
                "quants": {
                    "f16": {"weights": {"backbone": "f16", "dit": "f16"}, "session": {}},
                },
                "defaultQuant": "f16",
                "pipelineConfig": {
                    "maxTextLen": 17,
                    "maxCaptionLen": 19,
                    "speakerRows": 23,
                    "ditSymMax": 29,
                    "frameRate": 7,
                    "sampleRate": 31,
                    "hopLength": 37,
                    "codecHaloFrames": 3,
                    "latentDim": 41,
                    "speakerPatchSize": 5,
                    "speakerDim": 43,
                    "textDim": 47,
                    "captionDim": 53,
                    "timestepEmbedDim": 59,
                    "steps": 61,
                    "initScale": 0.111,
                    "cfgMinT": 0.25,
                    "cfgMaxT": 0.75,
                    "cfgScales": {"text": 1.5, "speaker": 2.5, "caption": 3.5},
                    "minSeconds": 0.5,
                    "maxSeconds": 4.0,
                    "speakerUncondMode": "mask",
                    "cfgGuidanceMode": "independent",
                },
            }
        },
    }


class TestIrodoriEntryPoint:
    """The load entry point the card points to — `fromPretrained` only."""

    def test_it_does_not_advertise_the_local_asset_entry_point(self) -> None:
        """`fromAssets` is not advertised (ruling of 2026-08-29).

        It can now read the split distribution too (X2-101), but it is a local-debugging surface
        that assumes the caller holds the bytes; the normal entry point for readers using HF is
        `fromPretrained`. Listing both would leave the reader to decide "which one to use".
        `fromPretrained` is checked as well so that the check does not pass when the whole Usage
        section is gone.
        """
        card = render_irodori_model_card(_irodori_manifest(), REPO)
        assert "fromAssets" not in card
        assert "await Pipeline.fromPretrained" in card

    def test_the_usage_snippet_declares_the_pipeline_with_await_using(self) -> None:
        """`irodori.Pipeline` has only `[Symbol.asyncDispose]` (a synchronous `using` fails at
        the declaration).

        A substring match passes for both `await using` and `using`, so the whole declaration
        line is compared.
        """
        lines = render_irodori_model_card(_irodori_manifest(), REPO).splitlines()
        assert "await using pipeline = await Pipeline.fromPretrained({" in lines
        assert not any(line.startswith("using pipeline") for line in lines)


class TestIrodoriUpstreamAttribution:
    """Attribution is looked up from the model name — the check stops a credit line that names
    the wrong version."""

    def test_v4_renders_its_own_upstream(self) -> None:
        card = render_irodori_model_card(_irodori_manifest("v4-small"), REPO)
        assert "Aratako/Irodori-TTS-v4-Small" in card
        assert "# Irodori-TTS v4 Small — Karume" in card
        assert "Irodori-TTS-v4.1-Small" not in card

    def test_v4_1_renders_its_own_upstream(self) -> None:
        card = render_irodori_model_card(_irodori_manifest("v4.1-small"), REPO)
        assert "Aratako/Irodori-TTS-v4.1-Small" in card
        assert "# Irodori-TTS v4.1 Small — Karume" in card
        assert "Irodori-TTS-v4-Small" not in card

    def test_unknown_model_name_fails_loudly(self) -> None:
        """A version missing from the table does not silently get another attribution (fail
        loudly)."""
        with pytest.raises(ValueError, match="IRODORI_UPSTREAMS"):
            render_irodori_model_card(_irodori_manifest("v9-small"), REPO)


class TestIrodoriHouseSentence:
    def test_the_title_is_followed_by_the_house_sentence(self) -> None:
        """One sentence over the three redistributed upstreams, each author named once."""
        lines = render_irodori_model_card(_irodori_manifest("v4-small"), REPO).splitlines()
        title = lines.index("# Irodori-TTS v4 Small — Karume")
        assert lines[title + 2] == (
            "This is an unofficial conversion of Aratako/Irodori-TTS-v4-Small,"
            " sbintuitions/modernbert-ja-310m and Aratako/Semantic-DACVAE-Japanese-32dim for"
            " Karume. It is not affiliated with or endorsed by Aratako and SB Intuitions."
        )


def _meanflow_manifest() -> dict[str, Any]:
    """The fake manifest of the MeanFlow student (its pipelineConfig declares the MeanFlow sampler
    and carries no RF knob — ADR 0150)."""
    manifest = _irodori_manifest("v4.1-small-mf")
    config = manifest["models"]["v4.1-small-mf"]["pipelineConfig"]
    for key in ("initScale", "cfgMinT", "cfgMaxT", "cfgScales"):
        del config[key]
    del config["speakerUncondMode"], config["cfgGuidanceMode"]
    config["flowParameterization"] = "meanflow"
    config["steps"] = 3
    return manifest


class TestIrodoriMeanFlowCard:
    """The card of the MeanFlow student says what the model is (ADR 0150)."""

    def test_the_card_names_the_student_its_revision_and_its_teacher(self) -> None:
        card = render_irodori_model_card(_meanflow_manifest(), REPO)

        assert "# Irodori-TTS v4.1 Small MF — Karume" in card
        assert (
            "[Aratako/Irodori-TTS-v4.1-Small-MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF)"
            " at revision `ccc78f5d480b6e51b69b2d5042a14c4da04fea6e`, licensed **MIT**"
        ) in card
        assert "distilled it from [Aratako/Irodori-TTS-v4.1-Small]" in card
        assert "MeanFlow-distilled DiT" in card

    def test_the_defaults_state_the_step_count_and_no_runtime_guidance(self) -> None:
        card = render_irodori_model_card(_meanflow_manifest(), REPO)

        assert "- **steps**: 3 MeanFlow steps" in card
        assert "- **guidance**: none at run time" in card
        assert "initScale" not in card
        assert "plus one per classifier-free-guidance branch" not in card

    def test_the_rf_card_keeps_its_wording(self) -> None:
        card = render_irodori_model_card(_irodori_manifest(), REPO)

        assert "rectified-flow DiT" in card
        assert "- **steps**: 61 Euler steps (`initScale` 0.111)" in card
        assert "at revision" not in card

    def test_the_meanflow_spelling_is_the_distributions(self) -> None:
        from irodori.card import IRODORI_MEANFLOW
        from irodori.distribution import IRODORI_FLOW_MEANFLOW

        assert IRODORI_MEANFLOW == IRODORI_FLOW_MEANFLOW
