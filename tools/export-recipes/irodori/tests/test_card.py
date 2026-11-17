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

from irodori.card import (
    IRODORI_SUPPORTED_PIPELINE,
    render_irodori_large_model_card,
    render_irodori_model_card,
)

#: The repo ID written into the usage snippet (dist passes it from the pipeline declaration).
REPO = "hdae/fake-repo"


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


def _container(*refs: dict[str, Any]) -> dict[str, Any]:
    """One dtype of `weights` (the `container` of `karume/6` — ADR 0109 decision 3).

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
        "format": "karume/6",
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


def _large_manifest(codec_repo: str | None = "hdae/karume-irodori-v4-small") -> dict[str, Any]:
    """The fake manifest of v4-Large: f16 / i8 on every component, i4 on the DiT pair, the codec
    referenced in another repository (`codec_repo`; `None` = stored here)."""
    manifest = _irodori_manifest("v4-large")
    model = manifest["models"]["v4-large"]
    names = ("backbone", "dit", "dit_context", "codec_decoder", "codec_encoder")
    weights: dict[str, Any] = {}
    for name in names:
        labels = ("f16", "i8", "i4") if name in ("dit", "dit_context") else ("f16", "i8")
        weights[name] = {}
        for label in labels:
            ref = _ref(f"v4-large/{name}/model.{label}-00001-of-00003.krm", 11, "c")
            if name.startswith("codec_") and codec_repo is not None:
                ref |= {"repo": codec_repo, "revision": "0" * 40}
            weights[name][label] = _container(ref)
    model["weights"] = weights
    model["quants"] = {
        "f16": {"weights": dict.fromkeys(names, "f16"), "session": {}},
        "i8": {"weights": dict.fromkeys(names, "i8"), "session": {}},
    }
    model["defaultQuant"] = "i8"
    return manifest


class TestIrodoriLargeCard:
    """The card of the v4-Large line: the Gemma terms as upstream states them, the T5Gemma 2 text
    encoder, the codec referenced in v4-small's repository, and no rare-token split."""

    def test_the_frontmatter_carries_upstreams_license_tag_and_the_text_encoder(self) -> None:
        card = render_irodori_large_model_card(_large_manifest(), REPO)
        front = card.split("---")[1]

        assert "license: gemma" in front
        assert "license_link" not in front
        assert "  - Aratako/Irodori-TTS-v4-Large" in front
        assert "  - google/t5gemma-2-1b-1b" in front
        assert "  - Aratako/Semantic-DACVAE-Japanese-32dim" in front

    def test_the_title_is_followed_by_the_house_sentence(self) -> None:
        lines = render_irodori_large_model_card(_large_manifest(), REPO).splitlines()
        title = lines.index("# Irodori-TTS v4 Large — Karume")
        assert lines[title + 2] == (
            "This is an unofficial conversion of Aratako/Irodori-TTS-v4-Large,"
            " google/t5gemma-2-1b-1b and Aratako/Semantic-DACVAE-Japanese-32dim for Karume. It is"
            " not affiliated with or endorsed by Aratako and Google."
        )

    def test_the_attribution_repeats_upstreams_license_statement(self) -> None:
        card = render_irodori_large_model_card(_large_manifest(), REPO)

        assert (
            "[Aratako/Irodori-TTS-v4-Large](https://huggingface.co/Aratako/Irodori-TTS-v4-Large)"
            " at revision `2e0c55428ce97268a507f1feeb2478f8d9148e8b`,"
        ) in card
        assert "[Gemma Terms of Use](https://ai.google.dev/gemma/terms)" in card
        assert (
            "[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)"
        ) in card
        assert "licensed **MIT**\n  (as of retrieval). This repository does not store it" in card
        assert "ModernBERT" not in card
        assert "licensed **MIT** (as of retrieval). It is redistributed here" not in card

    def test_a_codec_stored_here_is_said_to_be_redistributed_here(self) -> None:
        card = render_irodori_large_model_card(_large_manifest(codec_repo=None), REPO)

        assert "It is redistributed here in the container format as the" in card
        assert "does not store it" not in card

    def test_the_changes_follow_the_manifests_storage(self) -> None:
        card = render_irodori_large_model_card(_large_manifest(), REPO)

        assert "every component is stored as `f16` / `i8` series, and the DiT" in card
        assert "adds an `i4` series rounded with GPTQ calibration" in card
        assert "no `f32` series is distributed" in card

    def test_a_storage_label_on_only_some_components_fails(self) -> None:
        manifest = _large_manifest()
        model = manifest["models"]["v4-large"]
        model["weights"]["backbone"]["i4"] = model["weights"]["backbone"]["i8"]

        with pytest.raises(ValueError, match="neither on every component nor exactly"):
            render_irodori_large_model_card(manifest, REPO)

    def test_the_usage_does_not_offer_the_rare_token_split(self) -> None:
        large = render_irodori_large_model_card(_large_manifest(), REPO)
        small = render_irodori_model_card(_irodori_manifest(), REPO)

        assert "splitRareTokens: false" not in large
        assert "// splitRareTokens: false, // keep rare long words" in small
        assert "The tokenizer asset carries no rare-token split table" in large
        assert "`splitRareTokens` follows the tokenizer asset" not in large
        assert "`splitRareTokens` follows the tokenizer asset" in small

    def test_each_lines_card_refuses_the_other_lines_model(self) -> None:
        with pytest.raises(ValueError, match="IRODORI_UPSTREAMS"):
            render_irodori_model_card(_large_manifest(), REPO)
        with pytest.raises(ValueError, match="IRODORI_LARGE_UPSTREAMS"):
            render_irodori_large_model_card(_irodori_manifest("v4-small"), REPO)

    def test_the_component_spellings_are_the_distributions(self) -> None:
        from irodori.card import IRODORI_CODEC_COMPONENTS, IRODORI_DIT_COMPONENTS
        from irodori.distribution import IRODORI_CODEC_DIRS, IRODORI_DIT_ROLES

        assert tuple(IRODORI_CODEC_DIRS) == IRODORI_CODEC_COMPONENTS
        assert IRODORI_DIT_ROLES == IRODORI_DIT_COMPONENTS
