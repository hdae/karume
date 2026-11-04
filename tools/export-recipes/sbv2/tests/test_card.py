"""Model card rendering of the SBV2 distribution (`sbv2.card`).

The real distribution's `karume.json` is not used — the job here is to see that the values of a
fake manifest appear in the text as they are (= that not a single hand-written number is mixed
in), so building it with values that **differ** from the real ones detects more. For the same
reason the style table and the speaker table use different names and a different order from the
real FN4 weights (baking a table in would fail).

Since manifest v2 (`karume/2` — ADR 0041) a card describes several models of one repository. The
fake manifest defaults to the family shape (2 models, with shared files) so that rendering that
only works for a single model does not slip through.

Attribution is a per-family profile (`fn` / `jvnv`). The sections that look at what is derived from
the manifest need only `fn` (a profile affects only the attribution section and the frontmatter),
so the profile-specific facts and the detection of mix-ups are gathered in `TestSbv2CardProfiles`
at the end.

NOTE: only this test imports `siglip2.card` (the only code coupling between recipes) —
{@link TestSbv2PipelineGate.test_the_two_templates_do_not_answer_for_each_other} asserts that "2
**real templates** reject each other's manifest", and replacing the counterpart with a synthetic
one would break the claim. It has had the same subject since the SigLIP2 template lived in core;
the move only changed what it points at.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from sbv2.card import (
    SBV2_CARD_PROFILES,
    SBV2_DEMO_TEXT,
    SBV2_FN_PROFILE,
    SBV2_JVNV_PROFILE,
    SBV2_QUANT_ROUNDING,
    SBV2_SUPPORTED_PIPELINE,
    Sbv2CardProfile,
    render_sbv2_model_card,
)
from sbv2.distribution import SBV2_QUANT_ABBREVIATIONS, SBV2_QUANTS
from siglip2.card import SIGLIP2_SUPPORTED_PIPELINE, render_siglip2_model_card

#: The repository ID spelled in the usage snippet (dist passes it from the pipeline's declaration).
REPO = "hdae/fake-repo"


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


def _container(*refs: dict[str, Any]) -> dict[str, Any]:
    """One dtype of weights (the `container` of `karume/5` — ADR 0109 decision 3).

    `descriptor` is a field the card does not read, so only its shape follows the real thing (the
    lengths and sha256 of the 2 documents).
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


def _siglip2_manifest(model: str = "base") -> dict[str, Any]:
    """A minimal SigLIP2 manifest (the counterpart for looking at the template mix-up gate in both
    directions)."""
    return {
        "format": "karume/5",
        "generator": "karume/9.9.9",
        "defaultModel": model,
        "models": {
            model: {
                "pipeline": SIGLIP2_SUPPORTED_PIPELINE,
                "weights": {
                    "vision": {"f32": _container(_ref("v/model.f32-00001-of-00003.krm", 11, "c"))}
                },
                "assets": {},
                "quants": {"f32": {"weights": {"vision": "f32"}, "session": {}}},
                "defaultQuant": "f32",
                "pipelineConfig": {
                    "imageWidth": 64,
                    "imageHeight": 96,
                    "imageMean": [0.1, 0.2, 0.3],
                    "imageStd": [0.4, 0.5, 0.6],
                    "hiddenDim": 7,
                    "interpolation": "bilinear",
                },
            }
        },
    }


def _sbv2_manifest() -> dict[str, Any]:
    """A minimal SBV2 manifest that keeps the shape of ADR 0041 (fake values that do not overlap the
    real ones).

    `styles` is **not sorted by ID** — so that the card printing the manifest's order as is (no
    sorting in between) can be observed directly.
    """
    text_encoder = _ref("shared/text_encoder/model.i8-00001-of-00003.krm", 555, "e")
    tokenizer = _ref("shared/tokenizer/fake-tokenizer.json", 11, "3")
    return {
        "format": "karume/5",
        "generator": "karume/9.9.9",
        "defaultModel": "ZA",
        "models": {
            "ZA": {
                "pipeline": SBV2_SUPPORTED_PIPELINE,
                "weights": {
                    "text_encoder": {"i8": _container(text_encoder)},
                    "front": {
                        "f16": _container(_ref("ZA/front/model.f16-00001-of-00003.krm", 666, "f")),
                        "i8": _container(_ref("ZA/front/model.i8-00001-of-00003.krm", 777, "0")),
                    },
                    "voice": {
                        "f16": _container(_ref("ZA/voice/model.f16-00001-of-00003.krm", 888, "1")),
                        "i8": _container(_ref("ZA/voice/model.i8-00001-of-00003.krm", 999, "2")),
                    },
                },
                "assets": {
                    "tokenizer": tokenizer,
                    "symbols": _ref("shared/text/symbols.json", 22, "4"),
                    "style_vectors": _ref("ZA/styles/fake_styles.safetensors", 33, "5"),
                    "speaker_embeddings": _ref("ZA/speakers/fake_spk.safetensors", 44, "6"),
                },
                "quants": {
                    "f16": {
                        "weights": {"text_encoder": "i8", "front": "f16", "voice": "f16"},
                        "session": {},
                    },
                    "i8": {
                        "weights": {"text_encoder": "i8", "front": "i8", "voice": "i8"},
                        "session": {},
                    },
                    "i8-a8": {
                        "weights": {"text_encoder": "i8", "front": "i8", "voice": "i8"},
                        "session": {"linearCompute": "a8"},
                        "label": "Fake label",
                        "description": "A fake description.",
                    },
                },
                "defaultQuant": "i8-a8",
                "pipelineConfig": {
                    "styles": {"Shout": 2, "Calm": 0, "Whisper": 1},
                    "speakers": {"ZZ9": 0, "ZZ8": 1},
                    "defaults": {
                        "speaker": "ZZ8",
                        "style": "Whisper",
                        "styleWeight": 0.75,
                        "sdpRatio": 0.25,
                        "noiseScale": 0.65,
                        "noiseScaleW": 0.85,
                        "lengthScale": 1.5,
                    },
                },
            },
            "ZB": {
                "pipeline": SBV2_SUPPORTED_PIPELINE,
                "weights": {
                    "text_encoder": {"i8": _container(text_encoder)},
                    "front": {
                        "i8": _container(_ref("ZB/front/model.i8-00001-of-00003.krm", 100, "7"))
                    },
                    "voice": {
                        "i8": _container(_ref("ZB/voice/model.i8-00001-of-00003.krm", 200, "8"))
                    },
                },
                "assets": {
                    "tokenizer": tokenizer,
                    "symbols": _ref("shared/text/symbols.json", 22, "4"),
                    "style_vectors": _ref("ZB/styles/fake_styles.safetensors", 55, "9"),
                    "speaker_embeddings": _ref("ZB/speakers/fake_spk.safetensors", 66, "a"),
                },
                "quants": {
                    "i8": {
                        "weights": {"text_encoder": "i8", "front": "i8", "voice": "i8"},
                        "session": {},
                    }
                },
                "defaultQuant": "i8",
                "pipelineConfig": {
                    "styles": {"Flat": 0},
                    "speakers": {"YY1": 0},
                    "defaults": {"speaker": "YY1", "style": "Flat", "lengthScale": 1.0},
                },
            },
        },
    }


@pytest.fixture
def sbv2_card() -> str:
    """The card rendered with the `fn` profile (what is derived from the manifest does not depend
    on the profile)."""
    return render_sbv2_model_card(_sbv2_manifest(), REPO, SBV2_FN_PROFILE, SBV2_QUANT_ABBREVIATIONS)


class TestSbv2Frontmatter:
    def test_it_declares_the_fields_hf_reads(self, sbv2_card: str) -> None:
        head = sbv2_card.split("---")[1]
        assert "library_name: karume" in head
        assert "pipeline_tag: text-to-speech" in head
        assert "base_model: rufflet17/voice_models" in head
        # A quantized distribution, so not merge / finetune (the same ruling as Anima).
        assert "base_model_relation: quantized" in head
        assert "license: other" in head
        assert [line for line in head.splitlines() if line.startswith("  - ")] == [
            "  - text-to-speech",
            "  - webgpu",
            "  - japanese",
        ]


class TestSbv2Sections:
    def test_it_carries_every_section(self, sbv2_card: str) -> None:
        headings = [line for line in sbv2_card.splitlines() if line.startswith("## ")]
        assert headings == [
            "## What is this",
            "## Base weights and attribution",
            "## Models",
            "## Usage",
            "## Model: ZA",
            "## Model: ZB",
        ]

    def test_each_model_section_carries_the_tables_the_voice_needs(self, sbv2_card: str) -> None:
        _, _, rest = sbv2_card.partition("## Model: ZA")
        first, _, second = rest.partition("## Model: ZB")
        for part in (first, second):
            assert [line for line in part.splitlines() if line.startswith("### ")] == [
                "### Quants",
                "### Styles",
                "### Speakers",
                "### Defaults",
            ]

    def test_it_attributes_both_upstream_models(self, sbv2_card: str) -> None:
        """Attribution is a fact the manifest does not have — both where the voice comes from and
        the text encoder that is redistributed."""
        assert "rufflet17/voice_models" in sbv2_card
        assert "2.6.1-JP-Extra" in sbv2_card
        assert "ku-nlp/deberta-v2-large-japanese-char-wwm" in sbv2_card
        assert "cc-by-sa-4.0" in sbv2_card

    def test_it_shows_the_minimal_typescript_entry_point(self, sbv2_card: str) -> None:
        """Spells only the public surface that exists (`encodeWav` of `packages/models/mod.ts` +
        the 2 names of `packages/models/sbv2.ts` + the real signatures)."""
        assert "await Pipeline.fromPretrained({" in sbv2_card
        assert f'  repo: "{REPO}",' in sbv2_card
        assert '  // revision: "<full commit sha>",' in sbv2_card
        assert "@karume/models" in sbv2_card
        assert "using pipeline" in sbv2_card
        assert "encodeWav(audio.data, audio.sampleRate)" in sbv2_card
        assert "const audio = await pipeline.generate(utterance, {" in sbv2_card

    def test_the_usage_snippet_declares_the_pipeline_with_await_using(self, sbv2_card: str) -> None:
        """`sbv2.Pipeline` has only `[Symbol.asyncDispose]` — a synchronous `using` fails at the
        declaration."""
        lines = sbv2_card.splitlines()
        assert "await using pipeline = await Pipeline.fromPretrained({" in lines
        assert not any(line.startswith("using pipeline") for line in lines)

    def test_the_usage_snippet_hands_generate_an_analyzed_utterance(self, sbv2_card: str) -> None:
        """The 2-layer input of `generate(utterance, options)` (0.6.0 on) — text analysis is the
        caller's job.

        The form that puts raw text in the options (`generate({ text })`) does not type-check.
        """
        lines = sbv2_card.splitlines()
        assert 'import { encodeWav } from "jsr:@karume/models";' in lines
        assert 'import { Pipeline, toUtterance } from "jsr:@karume/models/sbv2";' in lines
        assert (
            "const utterance = toUtterance(analyzeWithWords(await getDictionary(),"
            f' "{SBV2_DEMO_TEXT}"));' in lines
        )
        assert not any(line.lstrip().startswith("text:") for line in lines)

    def test_it_does_not_promise_a_dictionary_the_package_does_not_fetch(
        self, sbv2_card: str
    ) -> None:
        """The package does not fetch a dictionary and has no `dictionary` option (analysis is the
        caller's)."""
        assert "`dictionary` option" not in sbv2_card
        assert "the pipeline fetches it" not in sbv2_card
        assert "belongs to the caller" in sbv2_card
        assert (
            "https://github.com/hdae/karume/tree/main/apps/playground/src/models/sbv2" in sbv2_card
        )
        assert "examples/sbv2" not in sbv2_card


class TestSbv2QuantRounding:
    """The note on **how the weights were rounded** that follows the quant table (the table does not
    show whether the same `i4` is GPTQ-calibrated or plain RTN)."""

    @staticmethod
    def _notes(card: str) -> str:
        _, _, rest = card.partition("## Model: ZA")
        _, _, notes = rest.partition("How the stored weights were rounded:")
        return notes.partition("### Styles")[0]

    def test_it_explains_only_the_quants_the_model_declares(self, sbv2_card: str) -> None:
        """Quants the distribution does not have are not explained (the same discipline as
        `_sbv2_knob`)."""
        notes = self._notes(sbv2_card)
        assert "- `i8` —" in notes
        assert "`i8+bert4`" not in notes
        assert "- `i4` —" not in notes

    def test_the_i8_note_names_the_granularity_the_implementation_uses(
        self, sbv2_card: str
    ) -> None:
        """The rounding is `karume.quantize.fake_quant_int8` = a scale **per output channel** (ADR
        0019).

        In the context of quantization "per tensor" is the established term for "one scale for the
        whole tensor", so calling it that would make the distribution claim a coarser method than
        the real one.
        """
        notes = self._notes(sbv2_card)
        assert "rounded per output channel (plain RTN)" in notes
        assert "per tensor" not in notes

    def test_every_shipped_seat_has_a_rounding_note(self) -> None:
        """A missing note cannot appear silently on the day a quant is added (`_sbv2_quants` skips
        keys that do not exist)."""
        assert set(SBV2_QUANT_ROUNDING) == set(SBV2_QUANTS)

    def test_a_model_declaring_every_seat_gets_a_note_for_every_seat(self) -> None:
        manifest = _sbv2_manifest()
        model = manifest["models"]["ZA"]
        for role, digit in (("text_encoder", "b"), ("front", "c"), ("voice", "d")):
            model["weights"][role]["i4"] = _container(_ref(f"ZA/{role}/model.i4.krm", 12, digit))
        model["quants"] = copy.deepcopy(dict(SBV2_QUANTS))
        model["defaultQuant"] = "i8"

        notes = self._notes(
            render_sbv2_model_card(manifest, REPO, SBV2_FN_PROFILE, SBV2_QUANT_ABBREVIATIONS)
        )

        for name in SBV2_QUANTS:
            assert f"- `{name}`" in notes, name

    def test_the_default_mark_follows_default_quant(self, sbv2_card: str) -> None:
        """The default mark comes from the manifest — baking it into the notes would make them
        disagree with the table alone when the default moves."""
        assert "- `i8-a8` (default) —" in self._notes(sbv2_card)
        assert "- `i8` (default)" not in self._notes(sbv2_card)

        manifest = _sbv2_manifest()
        manifest["models"]["ZA"]["defaultQuant"] = "i8"
        moved = render_sbv2_model_card(manifest, REPO, SBV2_FN_PROFILE, SBV2_QUANT_ABBREVIATIONS)
        assert "- `i8` (default) —" in self._notes(moved)


class TestSbv2Derivation:
    """MUST: the numbers, the download sizes, the quant table, the style table and the speaker table
    come from the manifest."""

    def test_each_seat_counts_the_files_it_declares(self, sbv2_card: str) -> None:
        """The table with one row per shard was retired (ruling of 2026-09-03) — the total of each
        quant goes into the Download column.

        `f16` = text_encoder 555 + front 666 + voice 888 + assets 110 = 2,219 B; for `i8` front /
        voice become 777 / 999, giving 2,441 B. Not a single weight path appears in the card.
        """
        _, _, rest = sbv2_card.partition("## Model: ZA")
        quants, _, _ = rest.partition("### Styles")

        assert "| 2.17 KiB (588 B shared; 110 B of assets, read on the host) |" in quants
        assert "| 2.38 KiB (588 B shared; 110 B of assets, read on the host) |" in quants
        for path in (
            "shared/text_encoder/model.i8-00001-of-00003.krm",
            "ZA/voice/model.f16-00001-of-00003.krm",
        ):
            assert path not in sbv2_card, path

    def test_it_takes_the_sizes_from_the_manifest(self, sbv2_card: str) -> None:
        manifest = _sbv2_manifest()
        manifest["models"]["ZA"]["weights"]["text_encoder"]["i8"]["container"]["parts"][0][
            "size"
        ] = 12345
        moved = render_sbv2_model_card(manifest, REPO, SBV2_FN_PROFILE, SBV2_QUANT_ABBREVIATIONS)
        assert "| 2.17 KiB (" in sbv2_card
        assert "| 2.17 KiB (" not in moved
        assert "| 13.7 KiB (" in moved

    def test_it_lists_every_style_with_its_row_id(self, sbv2_card: str) -> None:
        rows = [line for line in sbv2_card.splitlines() if line.startswith("| `")]
        assert "| `Shout` | 2 |" in rows
        assert "| `Calm` | 0 |" in rows
        assert "| `Whisper` | 1 |" in rows

    def test_it_keeps_the_style_table_in_the_manifests_own_order(self) -> None:
        """No sorting in between — changing the manifest's order changes the table's order too."""

        def style_rows(manifest: dict[str, Any]) -> list[str]:
            card = render_sbv2_model_card(manifest, REPO, SBV2_FN_PROFILE, SBV2_QUANT_ABBREVIATIONS)
            head, _, rest = card.partition("### Styles")
            assert head  # if the section disappeared, the extraction below would be meaningless
            return [line for line in rest.splitlines() if line.startswith("| `")]

        manifest = _sbv2_manifest()
        assert style_rows(manifest)[:3] == [
            "| `Shout` | 2 |",
            "| `Calm` | 0 |",
            "| `Whisper` | 1 |",
        ]
        manifest["models"]["ZA"]["pipelineConfig"]["styles"] = {
            "Calm": 0,
            "Whisper": 1,
            "Shout": 2,
        }
        assert style_rows(manifest)[:3] == [
            "| `Calm` | 0 |",
            "| `Whisper` | 1 |",
            "| `Shout` | 2 |",
        ]

    def test_it_renames_the_styles_when_the_manifest_does(self) -> None:
        """Names are not baked in either (a distribution of another ckpt shows other style
        names)."""
        manifest = _sbv2_manifest()
        del manifest["models"]["ZB"]
        manifest["models"]["ZA"]["pipelineConfig"]["styles"] = {"Angry": 0}
        manifest["models"]["ZA"]["pipelineConfig"]["defaults"]["style"] = "Angry"
        card = render_sbv2_model_card(manifest, REPO, SBV2_FN_PROFILE, SBV2_QUANT_ABBREVIATIONS)
        assert "| `Angry` | 0 |" in card
        for gone in ("Shout", "Calm", "Whisper"):
            assert gone not in card, gone

    def test_it_lists_every_speaker_with_its_row_id(self, sbv2_card: str) -> None:
        _, _, rest = sbv2_card.partition("### Speakers")
        rows = [line for line in rest.splitlines() if line.startswith("| `")]
        assert rows[:2] == ["| `ZZ9` | 0 |", "| `ZZ8` | 1 |"]

    def test_it_names_the_tables_the_ids_index(self, sbv2_card: str) -> None:
        """Where the row IDs resolve to is also taken from the manifest's file paths (the spelling
        is not copied)."""
        assert "`ZA/styles/fake_styles.safetensors`" in sbv2_card
        assert "`ZA/speakers/fake_spk.safetensors`" in sbv2_card
        assert "`ZB/styles/fake_styles.safetensors`" in sbv2_card

    def test_it_marks_exactly_the_default_quant_of_the_model(self, sbv2_card: str) -> None:
        _, _, rest = sbv2_card.partition("### Quants")
        table = rest.partition("### Styles")[0].partition("How the stored weights were rounded:")[0]
        rows = [line for line in table.splitlines()]
        default = [line for line in rows if "(default)" in line]
        assert len(default) == 1
        assert default[0].startswith("| `i8-a8` (default) |")
        assert (
            "| `f16` | — | 2.17 KiB (588 B shared; 110 B of assets, read on the host) |"
            " `text_encoder` = `i8` / `front` = `f16` / `voice` = `f16` | — |" in rows
        )

    def test_it_prints_the_presentation_fields_the_manifest_carries(self, sbv2_card: str) -> None:
        """The display fields appear in the card with the same strings (ADR 0075 decision 5)."""
        assert "**Fake label** — A fake description." in sbv2_card

    def test_it_explains_the_abbreviation_the_seat_names_use(self, sbv2_card: str) -> None:
        """The abbreviation mapping is **always** printed (ADR 0074 decision 4) — so it is readable
        which component `bert4` is about."""
        assert "In a quant name, `bert` is the `text_encoder` component." in sbv2_card

    def test_it_takes_every_default_knob_from_the_pipeline_config(self, sbv2_card: str) -> None:
        for line in (
            "- **speaker**: `ZZ8`",
            "- **style**: `Whisper`",
            "- **styleWeight**: 0.75",
            "- **sdpRatio**: 0.25",
            "- **noiseScale**: 0.65",
            "- **noiseScaleW**: 0.85",
            "- **lengthScale**: 1.5",
        ):
            assert line in sbv2_card, line

    def test_it_does_not_invent_knobs_the_manifest_left_out(self, sbv2_card: str) -> None:
        """The list of defaults is the manifest's `defaults` itself (no copy of the keys)."""
        _, _, second = sbv2_card.partition("## Model: ZB")
        assert "- **speaker**: `YY1`" in second
        assert "noiseScaleW" not in second

    def test_the_usage_snippet_follows_the_manifest(self, sbv2_card: str) -> None:
        """The snippet's model / quant / style also come from the manifest (it does not recommend a
        spelling that does not exist).

        The default is spelled out and **commented out** (the "just uncomment it" form of the
        ruling of 2026-08-12), and the selectable values are listed on the same line from the
        manifest (when voices / styles / quants are added, the list follows).
        """
        assert '  // model: "ZA", // default — available: ZA / ZB' in sbv2_card
        assert '  // quant: "i8-a8", // default — available: f16 / i8 / i8-a8' in sbv2_card
        assert '  // style: "Whisper", // default — available: Calm / Shout / Whisper' in sbv2_card
        assert '  // speaker: "ZZ8", // default — available: ZZ8 / ZZ9' in sbv2_card

    def test_the_usage_snippet_carries_every_delivery_knob_the_manifest_defines(
        self, sbv2_card: str
    ) -> None:
        """The remaining optionals of `generate()` are also written as comments with their defaults
        (the values come from the manifest)."""
        for line in (
            "  // styleWeight: 0.75, //",
            "  // lengthScale: 1.5, //",
            "  // sdpRatio: 0.25, //",
            "  // noiseScale: 0.65, //",
            "  // noiseScaleW: 0.85, //",
        ):
            assert line in sbv2_card, line

    def test_the_usage_snippet_skips_a_knob_the_manifest_left_out(self) -> None:
        """A knob not in `defaults` is not spelled (the same judgement as `### Defaults` — no
        invented values)."""
        manifest = _sbv2_manifest()
        manifest["defaultModel"] = "ZB"  # a model without noiseScaleW and the like
        usage = render_sbv2_model_card(
            manifest, REPO, SBV2_FN_PROFILE, SBV2_QUANT_ABBREVIATIONS
        ).partition("## Usage")[2]
        snippet = usage.partition("## Model:")[0]
        assert "  // lengthScale: 1.0, //" in snippet
        for absent in ("styleWeight", "sdpRatio", "noiseScale"):
            assert absent not in snippet, absent


class TestSbv2Determinism:
    def test_it_renders_the_same_bytes_for_the_same_manifest(self) -> None:
        first = render_sbv2_model_card(
            _sbv2_manifest(), REPO, SBV2_FN_PROFILE, SBV2_QUANT_ABBREVIATIONS
        )
        second = render_sbv2_model_card(
            json.loads(json.dumps(_sbv2_manifest())),
            REPO,
            SBV2_FN_PROFILE,
            SBV2_QUANT_ABBREVIATIONS,
        )
        assert first.encode("utf-8") == second.encode("utf-8")

    def test_it_does_not_mutate_the_manifest(self) -> None:
        manifest = _sbv2_manifest()
        before = copy.deepcopy(manifest)
        render_sbv2_model_card(manifest, REPO, SBV2_FN_PROFILE, SBV2_QUANT_ABBREVIATIONS)
        assert manifest == before


class TestSbv2PipelineGate:
    def test_it_refuses_a_pipeline_it_does_not_describe(self) -> None:
        manifest = _sbv2_manifest()
        manifest["models"]["ZA"]["pipeline"] = SIGLIP2_SUPPORTED_PIPELINE
        with pytest.raises(ValueError, match=SIGLIP2_SUPPORTED_PIPELINE):
            render_sbv2_model_card(manifest, REPO, SBV2_FN_PROFILE, SBV2_QUANT_ABBREVIATIONS)

    def test_the_two_templates_do_not_answer_for_each_other(self) -> None:
        """A template is pipeline-specific — a mix-up gives "the table matches but the description
        is of another model"."""
        with pytest.raises(ValueError, match=SBV2_SUPPORTED_PIPELINE):
            render_siglip2_model_card(_sbv2_manifest(), REPO)
        with pytest.raises(ValueError, match=SIGLIP2_SUPPORTED_PIPELINE):
            render_sbv2_model_card(
                _siglip2_manifest(), REPO, SBV2_FN_PROFILE, SBV2_QUANT_ABBREVIATIONS
            )


# ---- Attribution profiles (fn / jvnv) --------------------------------------


def _profile_card(profile: Sbv2CardProfile) -> str:
    return render_sbv2_model_card(_sbv2_manifest(), REPO, profile, SBV2_QUANT_ABBREVIATIONS)


class TestSbv2CardProfiles:
    """The section that checks that attribution (source, license, citation) is the only part that
    differs per family.

    A wrong attribution happens while the tables and the usage stay correct, so nobody can notice
    until it has been shipped — it observes in both directions that "not a single word of the other
    family's facts is mixed in".
    """

    def test_it_offers_exactly_the_families_it_can_attribute(self) -> None:
        assert sorted(SBV2_CARD_PROFILES) == ["fn", "jvnv"]
        assert SBV2_CARD_PROFILES["fn"] is SBV2_FN_PROFILE
        assert SBV2_CARD_PROFILES["jvnv"] is SBV2_JVNV_PROFILE

    def test_it_refuses_to_draw_a_card_without_naming_the_family(self) -> None:
        """There is no default profile — if it could be rendered by omission, wrong attribution
        would silently come back."""
        with pytest.raises(TypeError):
            render_sbv2_model_card(_sbv2_manifest(), REPO)  # type: ignore[call-arg]

    @pytest.mark.parametrize("name", sorted(SBV2_CARD_PROFILES))
    def test_every_profile_keeps_the_same_skeleton(self, name: str) -> None:
        """A profile moves only the frontmatter and the attribution section (the section order and
        the tables keep the same shape)."""
        card = _profile_card(SBV2_CARD_PROFILES[name])
        assert [line for line in card.splitlines() if line.startswith("## ")] == [
            "## What is this",
            "## Base weights and attribution",
            "## Models",
            "## Usage",
            "## Model: ZA",
            "## Model: ZB",
        ]
        assert "| `Whisper` | 1 |" in card
        assert '  repo: "hdae/fake-repo",' in card

    @pytest.mark.parametrize("name", sorted(SBV2_CARD_PROFILES))
    def test_every_profile_titles_the_card_after_its_own_family(self, name: str) -> None:
        card = _profile_card(SBV2_CARD_PROFILES[name])
        assert f"\n# {SBV2_CARD_PROFILES[name].title}\n" in card

    def test_the_jvnv_frontmatter_declares_the_license_hf_can_resolve(self) -> None:
        """A standard SPDX tag, so `license_name` / `license_link` are **not listed**."""
        head = _profile_card(SBV2_JVNV_PROFILE).split("---")[1]
        assert "license: cc-by-sa-4.0" in head
        assert "license_name" not in head
        assert "license_link" not in head

    def test_the_jvnv_frontmatter_lists_both_redistributed_upstreams(self) -> None:
        """What is redistributed is the voice and the text encoder — both with the relation
        `quantized`."""
        head = _profile_card(SBV2_JVNV_PROFILE).split("---")[1]
        assert "base_model:\n" in head
        assert "  - litagin/style_bert_vits2_jvnv\n" in head
        assert "  - ku-nlp/deberta-v2-large-japanese-char-wwm\n" in head
        assert "base_model_relation: quantized" in head

    def test_the_jvnv_attribution_names_every_source_directory(self) -> None:
        card = _profile_card(SBV2_JVNV_PROFILE)
        for directory in ("jvnv-F1-jp/", "jvnv-F2-jp/", "jvnv-M1-jp/", "jvnv-M2-jp/"):
            assert f"`{directory}`" in card, directory
        assert "`version: 2.0-JP-Extra`" in card

    def test_the_jvnv_attribution_carries_what_cc_by_sa_requires(self) -> None:
        """BY = credit, source, license URL and a notice of **the changes made** / SA = the same
        license."""
        card = _profile_card(SBV2_JVNV_PROFILE)
        for required in (
            "https://creativecommons.org/licenses/by-sa/4.0/",
            "credit the authors",
            "the license URL",
            "**converted to another format and quantized**",
            "license any derivative work under CC BY-SA 4.0",
            "no further",
            "There is no NonCommercial and no NoDerivatives clause",
        ):
            assert required in card, required

    def test_the_jvnv_attribution_cites_the_corpus_and_the_training_implementation(
        self,
    ) -> None:
        card = _profile_card(SBV2_JVNV_PROFILE)
        for required in (
            "Detai Xin",
            "Hiroshi Saruwatari",
            "JVNV: A Corpus of Japanese Emotional Speech",
            "https://arxiv.org/abs/2310.06072",
            "https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus",
            "https://github.com/litagin02/Style-Bert-VITS2",
            "AGPL-3.0",
            "Karume is a separate implementation and does not contain the original",
        ):
            assert required in card, required

    def test_neither_family_carries_the_others_attribution(self) -> None:
        """Not a single word of one family's facts leaks into the other (observing this in both
        directions is the core of catching wrong attribution)."""
        fn = _profile_card(SBV2_FN_PROFILE)
        jvnv = _profile_card(SBV2_JVNV_PROFILE)
        for leaked in ("jvnv", "litagin", "2.0-JP-Extra", "CC BY-SA 4.0", "arxiv"):
            assert leaked not in fn, leaked
        for leaked in ("rufflet17", "2.6.1-JP-Extra", "`FN/`", "license: other"):
            assert leaked not in jvnv, leaked

    @pytest.mark.parametrize("name", sorted(SBV2_CARD_PROFILES))
    def test_every_profile_renders_the_same_bytes_for_the_same_manifest(self, name: str) -> None:
        profile = SBV2_CARD_PROFILES[name]
        first = render_sbv2_model_card(_sbv2_manifest(), REPO, profile, SBV2_QUANT_ABBREVIATIONS)
        second = render_sbv2_model_card(
            json.loads(json.dumps(_sbv2_manifest())), REPO, profile, SBV2_QUANT_ABBREVIATIONS
        )
        assert first.encode("utf-8") == second.encode("utf-8")

    @pytest.mark.parametrize("name", sorted(SBV2_CARD_PROFILES))
    def test_no_profile_mutates_the_manifest(self, name: str) -> None:
        manifest = _sbv2_manifest()
        before = copy.deepcopy(manifest)
        render_sbv2_model_card(manifest, REPO, SBV2_CARD_PROFILES[name], SBV2_QUANT_ABBREVIATIONS)
        assert manifest == before


class TestSbv2EntryPoint:
    """The load entry point the card points to — only `fromPretrained`."""

    @pytest.mark.parametrize("name", sorted(SBV2_CARD_PROFILES))
    def test_it_does_not_advertise_the_local_asset_entry_point(self, name: str) -> None:
        """`fromAssets` is not advertised (ruling of 2026-08-29).

        The split distribution can be read too (X2-101), but that surface is for local debugging
        and assumes you hold the bytes yourself; the ordinary entry point for a reader using HF is
        `fromPretrained`. Listing both would leave the reader to decide "which one to use".
        `fromPretrained` is checked as well so that this does not become a gate that passes even
        when the whole Usage disappears. The Usage section is shared by the profiles, so every
        profile is checked, including the case of advice appearing in only one profile.
        """
        card = _profile_card(SBV2_CARD_PROFILES[name])
        assert "fromAssets" not in card
        assert "await Pipeline.fromPretrained" in card


class TestSbv2HouseSentence:
    @pytest.mark.parametrize(
        ("profile", "sentence"),
        [
            (
                SBV2_JVNV_PROFILE,
                "This is an unofficial conversion of litagin/style_bert_vits2_jvnv and"
                " ku-nlp/deberta-v2-large-japanese-char-wwm for Karume. It is not affiliated with"
                " or endorsed by litagin and ku-nlp.",
            ),
            (
                SBV2_FN_PROFILE,
                "This is an unofficial conversion of rufflet17/voice_models for Karume. It is not"
                " affiliated with or endorsed by rufflet17.",
            ),
        ],
    )
    def test_the_title_is_followed_by_the_house_sentence(
        self, profile: Sbv2CardProfile, sentence: str
    ) -> None:
        lines = _profile_card(profile).splitlines()
        assert lines[lines.index(f"# {profile.title}") + 2] == sentence
