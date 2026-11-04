"""Rendering the model card of the QAT distribution (`gemma4_qat.distribution.render_card`).

The real distribution's `karume.json` is not used — the job here is to see that the fake
manifest's values show up in the text as they are (= not a single hand-written number mixed in),
so building it with values **different** from the real ones detects more.

What lives here are the checks only the card side has — the template's pipeline specificity, the
attribution naming the upstream mobile QAT checkpoint, and the text's claim "keeping the fixed
values is never called requantization".
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import pytest

from gemma4_qat import distribution
from gemma4_qat.config import CHECKPOINTS, SOURCES, checkpoint_name, checkpoint_repo
from gemma4_qat.distribution import NOTICE, QAT_LICENSE_LINK, render_card

#: The repository ID spelled in the usage snippet (dist passes it from the pipeline's
#: declaration).
REPO = "hdae/fake-repo"

#: Another family's pipeline contract (**the spelling held as is** — so no code coupling between
#: recipes is created).
FOREIGN_PIPELINE = "gemma4/1"

QAT_PIPELINE = "gemma4-qat/1"

#: The quant's display field (**a spelling that does not overlap the real one** — if hand-written
#: text were mixed in, it would not show in the table).
FAKE_QUANT_LABEL = "Fake fixed seat"
FAKE_QUANT_DESCRIPTION = "Only this sentence may appear in the quant table."


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


def _container(*refs: dict[str, Any]) -> dict[str, Any]:
    """One dtype of weights (the `container` of `karume/5` — ADR 0109 decision 3).

    The card does not read `descriptor`, so only its shape follows the real one (the lengths and
    sha256 of the two documents).
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


def _qat_manifest(model: str = "e2b") -> dict[str, Any]:
    """The minimal QAT manifest (fake values that do not overlap the real ones)."""
    return {
        "format": "karume/5",
        "generator": "karume/9.9.9",
        "defaultModel": model,
        "models": {
            model: {
                "pipeline": QAT_PIPELINE,
                "weights": {
                    "model": {"i4": _container(_ref("model/model.i4-00001-of-00003.krm", 13, "a"))}
                },
                "assets": {"ple_index": _ref("ple/ple.json", 7, "b")},
                "quants": {
                    "i4": {
                        "weights": {"model": "i4"},
                        "session": {},
                        "label": FAKE_QUANT_LABEL,
                        "description": FAKE_QUANT_DESCRIPTION,
                    }
                },
                "defaultQuant": "i4",
                "pipelineConfig": {"chunkLength": 3, "capacity": 5, "maxPosition": 7},
            }
        },
    }


class TestQatCardGate:
    def test_it_refuses_a_pipeline_it_does_not_describe(self) -> None:
        manifest = _qat_manifest()
        manifest["models"]["e2b"]["pipeline"] = FOREIGN_PIPELINE
        with pytest.raises(ValueError, match=QAT_PIPELINE):
            render_card(manifest, REPO)

    def test_it_refuses_a_model_it_cannot_attribute(self) -> None:
        """Rendering with a model name not in the attribution table would silently ship a
        redistribution that names no upstream."""
        with pytest.raises(ValueError, match="unsupported"):
            render_card(_qat_manifest("12b"), REPO)

    def test_it_renders_the_same_bytes_for_the_same_manifest(self) -> None:
        manifest = _qat_manifest()
        before = copy.deepcopy(manifest)

        assert render_card(manifest, REPO) == render_card(manifest, REPO)
        assert manifest == before


class TestQatCardBody:
    def test_it_names_the_upstream_mobile_checkpoint(self) -> None:
        card = render_card(_qat_manifest(), REPO)

        assert f"google/{checkpoint_name('e2b')}" in card
        assert "gemma-4-E2B-it-qat-mobile-transformers" in card

    def test_it_does_not_call_preserved_integers_a_requantization(self) -> None:
        """Keeping the fixed values is never called requantization (the claim of `render_card`'s
        docstring)."""
        card = render_card(_qat_manifest(), REPO)

        assert "re-quantized" not in card
        assert "Fixed integers and scales are preserved" in card

    def test_it_takes_the_quant_presentation_from_the_manifest(self) -> None:
        """The presets' descriptions are the manifest's values as they are (the card side keeps no
        copy)."""
        card = render_card(_qat_manifest(), REPO)

        assert FAKE_QUANT_LABEL in card
        assert FAKE_QUANT_DESCRIPTION in card


def _two_model_manifest() -> dict[str, Any]:
    """Both models in one manifest (the published form), `e2b` the default."""
    manifest = _qat_manifest("e2b")
    manifest["models"]["e4b"] = copy.deepcopy(_qat_manifest("e4b")["models"]["e4b"])
    return manifest


def _attribution(card: str) -> str:
    return card.split("## Base weights and attribution")[1].split("\n## ")[0]


class TestQatCardAttribution:
    def test_it_names_each_models_upstream_at_its_pinned_commit(self) -> None:
        """Release runbook §1: the attribution matches the upstream per revision."""
        attribution = _attribution(render_card(_two_model_manifest(), REPO))

        for model in ("e2b", "e4b"):
            bullet = attribution.split(f"- **`{model}`**")[1].split("\n- ")[0]
            assert (
                f"[{checkpoint_repo(model)}](https://huggingface.co/{checkpoint_repo(model)})"
                in (bullet)
            )
            pins = SOURCES[model].pins
            assert f"at commit `{pins.weights.revision}`" in bullet
            tokenizer = pins.tokenizer.repo
            assert f"[{tokenizer}](https://huggingface.co/{tokenizer}) at commit" in bullet
            assert f"at commit `{pins.tokenizer.revision}`" in bullet
            assert "licensed **Apache 2.0**" in bullet
            assert "a verbatim copy is in `LICENSE.md`" in bullet

    def test_it_attributes_only_the_models_it_carries(self) -> None:
        """A model not in the manifest is not redistributed, so it is not named as a source."""
        card = render_card(_qat_manifest("e4b"), REPO)
        front_matter = card.split("---")[1]

        assert checkpoint_repo("e2b") not in card
        for pin in SOURCES["e2b"].pins.by_role().values():
            assert pin.revision not in card
        assert f"  - {SOURCES['e2b'].pins.tokenizer.repo}\n" not in front_matter

    def test_the_front_matter_points_to_the_licence_the_upstream_names(self) -> None:
        front_matter = render_card(_two_model_manifest(), REPO).split("---")[1]

        assert "license: apache-2.0" in front_matter
        assert f"license_link: {QAT_LICENSE_LINK}" in front_matter
        assert "base_model_relation: quantized" in front_matter

    def test_the_front_matter_lists_each_base_model_before_its_mobile_checkpoint(self) -> None:
        """Every redistributed upstream is a `base_model`: per model the regular `-it` repository
        (the tokenizer's source, and the model the mobile checkpoint is a QAT form of), then the
        mobile checkpoint — the q4_0 QAT line's card lists them in the same order."""
        front_matter = render_card(_two_model_manifest(), REPO).split("---")[1]

        listed = "".join(
            f"  - {SOURCES[model].pins.tokenizer.repo}\n  - {checkpoint_repo(model)}\n"
            for model in ("e2b", "e4b")
        )
        assert f"base_model:\n{listed}" in front_matter
        assert SOURCES["e2b"].pins.tokenizer.repo == "google/gemma-4-E2B-it"
        assert SOURCES["e4b"].pins.tokenizer.repo == "google/gemma-4-E4B-it"

    def test_it_states_the_changes(self) -> None:
        attribution = _attribution(render_card(_qat_manifest(), REPO))

        assert "Apache 2.0 §4(b)" in attribution
        assert "No retraining and no fine-tuning." in attribution

    def test_it_keeps_the_limitations_of_the_conversion(self) -> None:
        card = render_card(_qat_manifest(), REPO)

        assert "Experimental text-only conversion" in card
        assert "static range quantization (SRQ)" in card
        assert "MTP is not supported." in card

    def test_it_refuses_a_pin_whose_licence_it_does_not_attribute(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The attribution sentence says Apache 2.0; a pin under another licence must not render."""
        moved = {**SOURCES, "e2b": dataclasses.replace(SOURCES["e2b"], license="other")}
        monkeypatch.setattr(distribution, "SOURCES", moved)

        with pytest.raises(ValueError, match="other"):
            render_card(_qat_manifest(), REPO)


class TestQatNotice:
    def test_it_names_every_upstream_checkpoint_and_the_licence(self) -> None:
        for model in CHECKPOINTS:
            assert f"`{checkpoint_repo(model)}`" in NOTICE
            assert f"`{SOURCES[model].pins.tokenizer.repo}`" in NOTICE
        assert "Apache License, Version 2.0" in NOTICE
        assert "`LICENSE.md`" in NOTICE

    def test_it_lists_the_changes_without_calling_them_a_requantization(self) -> None:
        assert "were not requantized" in NOTICE
        assert "re-quantized" not in NOTICE
        assert "No retraining and no fine-tuning were performed." in NOTICE

    def test_it_is_the_notice_the_pipeline_ships(self) -> None:
        assert distribution.PIPELINE.root_files["NOTICE.md"] == NOTICE
