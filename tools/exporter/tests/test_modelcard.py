"""モデルカードの描画（`karume.modelcard`）。

実物の `models/anima-turbo/karume.json` は使わない — 偽 manifest の値がそのまま本文に出ること
（＝手書きの数値が 1 つも混ざっていないこと）を見るのがここの仕事なので、実物と**違う**値で
組んだほうが検出力が高い。SBV2 のスタイル表・話者表も同じ理由で実重み FN4 と別の名前・別の
並びにしてある（表を焼き込んでいれば落ちる）。
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from karume.modelcard import (
    HF_REPO,
    SBV2_HF_REPO,
    SBV2_SUPPORTED_PIPELINE,
    SUPPORTED_PIPELINE,
    render_model_card,
    render_sbv2_model_card,
)


def _manifest() -> dict[str, Any]:
    """ADR 0038 の形を保った最小の manifest（値は実物と重ならない偽値）。"""
    return {
        "format": "karume/1",
        "generator": "karume/9.9.9",
        "pipeline": SUPPORTED_PIPELINE,
        "components": {
            "text_encoder": {
                "file": {"path": "text_encoder/model.safetensors", "size": 111, "sha256": "a" * 64}
            },
            "transformer": {
                "variants": {
                    "f16": {
                        "file": {
                            "path": "transformer/model.f16.safetensors",
                            "size": 222,
                            "sha256": "b" * 64,
                        },
                        "extras": {
                            "rope_base": {
                                "path": "transformer/rope_base.safetensors",
                                "size": 333,
                                "sha256": "c" * 64,
                            }
                        },
                    },
                    "i8": {
                        "file": {
                            "path": "transformer/model.i8.safetensors",
                            "size": 444,
                            "sha256": "d" * 64,
                        },
                        "extras": {
                            "rope_base": {
                                "path": "transformer/rope_base.safetensors",
                                "size": 333,
                                "sha256": "c" * 64,
                            }
                        },
                    },
                }
            },
        },
        "presets": {
            "f16": {"weights": {"transformer": "f16"}, "session": {}},
            "w8a8": {"weights": {"transformer": "i8"}, "session": {"linearCompute": "i8a8"}},
            "f16-c16": {
                "weights": {"transformer": "f16"},
                "session": {"linearCompute": "f16"},
                "gpuFeatures": {"shaderF16": True},
            },
        },
        "defaultPreset": "w8a8",
        "pipelineConfig": {
            "scheduler": {"shift": 3, "numTrainTimesteps": 1000},
            "defaults": {
                "steps": 7,
                "guidanceScale": 1,
                "resolution": {"width": 640, "height": 384},
                "negativePrompt": "ネガティブの偽値",
            },
        },
    }


@pytest.fixture
def card() -> str:
    return render_model_card(_manifest())


class TestFrontmatter:
    def test_it_opens_with_a_yaml_block(self, card: str) -> None:
        lines = card.splitlines()
        assert lines[0] == "---"
        assert lines.index("---", 1) > 1

    def test_it_declares_the_fields_hf_reads(self, card: str) -> None:
        head = card.split("---")[1]
        assert "pipeline_tag: text-to-image" in head
        assert "base_model: circlestone-labs/Anima-Base-v1.0-Diffusers" in head
        assert "base_model_relation: quantized" in head
        assert "license: other" in head
        assert "license_name: circlestone-labs-non-commercial-license" in head
        assert "library_name: karume" in head
        assert [line for line in head.splitlines() if line.startswith("  - ")] == [
            "  - text-to-image",
            "  - webgpu",
        ]


class TestSections:
    def test_it_carries_every_section(self, card: str) -> None:
        headings = [line for line in card.splitlines() if line.startswith("## ")]
        assert headings == [
            "## What is this",
            "## Baked-in LoRA",
            "## Files",
            "## Presets",
            "## Usage",
            "## Defaults",
        ]

    def test_it_names_the_merged_lora_with_its_source_and_hash(self, card: str) -> None:
        assert "Anima Turbo LoRA v0.2" in card
        assert "circlestone_labs" in card
        assert "https://civitai.com/models/2560840?modelVersionId=2979642" in card
        assert "1b55e40bdb1d0e5a78cb498f245fccfdaae97823265db957d2aabdcf4cd3caf1" in card

    def test_it_shows_the_minimal_typescript_entry_point(self, card: str) -> None:
        assert f'AnimaPipeline.fromPretrained("{HF_REPO}")' in card
        assert "@karume/models" in card
        assert "using pipeline" in card


class TestDerivation:
    """MUST: 数値・ファイル一覧・preset 表は manifest 由来（手書きが混ざっていない）。"""

    def test_it_lists_every_declared_path_once(self, card: str) -> None:
        for path in (
            "text_encoder/model.safetensors",
            "transformer/model.f16.safetensors",
            "transformer/model.i8.safetensors",
            "transformer/rope_base.safetensors",
        ):
            assert card.count(f"`{path}`") == 1, path

    def test_it_folds_a_shared_extra_into_one_row_naming_both_variants(self, card: str) -> None:
        row = next(line for line in card.splitlines() if "rope_base.safetensors`" in line)
        assert "`transformer.rope_base`" in row
        assert "f16 / i8" in row

    def test_it_takes_the_sizes_from_the_manifest(self, card: str) -> None:
        manifest = _manifest()
        manifest["components"]["text_encoder"]["file"]["size"] = 999
        moved = render_model_card(manifest)
        assert "111 B" in card
        assert "111 B" not in moved
        assert "999 B" in moved

    def test_it_marks_exactly_the_default_preset_beside_its_name(self, card: str) -> None:
        rows = [line for line in card.splitlines() if line.startswith("| `")]
        default = [line for line in rows if "(default)" in line]
        assert len(default) == 1
        assert default[0].startswith("| `w8a8` (default) |")
        # 表に「既定」列を持たない（印は名前の横だけ — 列が空欄で並ぶ形にしない）。
        assert "| Preset | Weights | Compute |" in card

    def test_it_carries_every_preset_with_its_session_knobs(self, card: str) -> None:
        assert "| `f16` | `transformer` = `f16` | — |" in card
        assert "`linearCompute` = `i8a8`" in card
        assert "requires `shaderF16`" in card

    def test_it_takes_the_defaults_from_the_pipeline_config(self, card: str) -> None:
        assert "- **steps**: 7" in card
        assert "- **resolution**: 640 × 384" in card
        assert "ネガティブの偽値" in card

    def test_it_only_warns_about_the_unused_negative_prompt_at_guidance_one(
        self, card: str
    ) -> None:
        assert "the negative prompt is not used" in card
        manifest = _manifest()
        manifest["pipelineConfig"]["defaults"]["guidanceScale"] = 4
        assert "the negative prompt is not used" not in render_model_card(manifest)


class TestDeterminism:
    def test_it_renders_the_same_bytes_for_the_same_manifest(self) -> None:
        first = render_model_card(_manifest())
        second = render_model_card(json.loads(json.dumps(_manifest())))
        assert first.encode("utf-8") == second.encode("utf-8")

    def test_it_does_not_mutate_the_manifest(self) -> None:
        manifest = _manifest()
        before = copy.deepcopy(manifest)
        render_model_card(manifest)
        assert manifest == before


class TestPipelineGate:
    def test_it_refuses_a_pipeline_it_does_not_describe(self) -> None:
        manifest = _manifest()
        manifest["pipeline"] = "sbv2/1"
        with pytest.raises(ValueError, match="sbv2/1"):
            render_model_card(manifest)


# ---- SBV2（text-to-speech）---------------------------------------------------


def _sbv2_manifest() -> dict[str, Any]:
    """ADR 0038 の形を保った最小の SBV2 manifest（値は実物と重ならない偽値）。

    `styles` は **ID の昇順に並べない** — カードが manifest の並びをそのまま出すこと
    （並べ替えを挟んでいないこと）を、そのまま観測できるようにするため。
    """
    return {
        "format": "karume/1",
        "generator": "karume/9.9.9",
        "pipeline": SBV2_SUPPORTED_PIPELINE,
        "components": {
            "text_encoder": {
                "file": {
                    "path": "text_encoder/model.i8.safetensors",
                    "size": 555,
                    "sha256": "e" * 64,
                }
            },
            "front": {
                "variants": {
                    "f16": {
                        "file": {
                            "path": "front/model.f16.safetensors",
                            "size": 666,
                            "sha256": "f" * 64,
                        }
                    },
                    "i8": {
                        "file": {
                            "path": "front/model.i8.safetensors",
                            "size": 777,
                            "sha256": "0" * 64,
                        }
                    },
                }
            },
            "voice": {
                "variants": {
                    "f16": {
                        "file": {
                            "path": "voice/model.f16.safetensors",
                            "size": 888,
                            "sha256": "1" * 64,
                        }
                    },
                    "i8": {
                        "file": {
                            "path": "voice/model.i8.safetensors",
                            "size": 999,
                            "sha256": "2" * 64,
                        }
                    },
                }
            },
            "tokenizer": {
                "file": {"path": "tokenizer/fake-tokenizer.json", "size": 11, "sha256": "3" * 64}
            },
            "symbols": {"file": {"path": "text/symbols.json", "size": 22, "sha256": "4" * 64}},
            "style_vectors": {
                "file": {"path": "styles/fake_styles.safetensors", "size": 33, "sha256": "5" * 64}
            },
            "speaker_embeddings": {
                "file": {"path": "speakers/fake_spk.safetensors", "size": 44, "sha256": "6" * 64}
            },
        },
        "presets": {
            "f16": {"weights": {"front": "f16", "voice": "f16"}, "session": {}},
            "w8": {"weights": {"front": "i8", "voice": "i8"}, "session": {}},
            "w8a8": {
                "weights": {"front": "i8", "voice": "i8"},
                "session": {"linearCompute": "i8a8"},
            },
        },
        "defaultPreset": "w8a8",
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
    }


@pytest.fixture
def sbv2_card() -> str:
    return render_sbv2_model_card(_sbv2_manifest())


class TestSbv2Frontmatter:
    def test_it_declares_the_fields_hf_reads(self, sbv2_card: str) -> None:
        head = sbv2_card.split("---")[1]
        assert "library_name: karume" in head
        assert "pipeline_tag: text-to-speech" in head
        assert "base_model: rufflet17/voice_models" in head
        # 量子化した配布形なので merge / finetune ではない（Anima と同じ裁定）。
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
            "## Files",
            "## Presets",
            "## Styles",
            "## Speakers",
            "## Usage",
            "## Defaults",
        ]

    def test_it_attributes_both_upstream_models(self, sbv2_card: str) -> None:
        """帰属は manifest に無い事実 — 声の出所と、再配布する text encoder の両方。"""
        assert "rufflet17/voice_models" in sbv2_card
        assert "2.6.1-JP-Extra" in sbv2_card
        assert "ku-nlp/deberta-v2-large-japanese-char-wwm" in sbv2_card
        assert "cc-by-sa-4.0" in sbv2_card

    def test_it_shows_the_minimal_typescript_entry_point(self, sbv2_card: str) -> None:
        """実在する公開面だけを綴る（`packages/models/mod.ts` の 2 名 + 実シグネチャ）。"""
        assert f'Sbv2Pipeline.fromPretrained("{SBV2_HF_REPO}")' in sbv2_card
        assert "@karume/models" in sbv2_card
        assert "using pipeline" in sbv2_card
        assert "encodeWav(audio.data, audio.sampleRate)" in sbv2_card
        assert "const audio = await pipeline.generate({" in sbv2_card


class TestSbv2Derivation:
    """MUST: 数値・ファイル一覧・preset 表・スタイル表・話者表は manifest 由来。"""

    def test_the_file_table_lists_every_declared_path_once(self, sbv2_card: str) -> None:
        """表の行は宣言されたファイルと 1 対 1（宣言外の行も、重複した行も無い）。"""
        _, _, rest = sbv2_card.partition("## Files")
        files, _, _ = rest.partition("## Presets")
        declared = [
            entry["file"]["path"]
            for component in _sbv2_manifest()["components"].values()
            for entry in (
                component["variants"].values() if "variants" in component else (component,)
            )
        ]
        rows = [line for line in files.splitlines() if line.startswith("| `")]
        assert len(rows) == len(declared)
        for path in declared:
            assert files.count(f"`{path}`") == 1, path

    def test_it_takes_the_sizes_from_the_manifest(self, sbv2_card: str) -> None:
        manifest = _sbv2_manifest()
        manifest["components"]["text_encoder"]["file"]["size"] = 12345
        moved = render_sbv2_model_card(manifest)
        assert "555 B" in sbv2_card
        assert "555 B" not in moved
        assert "12,345 B" in moved

    def test_it_lists_every_style_with_its_row_id(self, sbv2_card: str) -> None:
        rows = [line for line in sbv2_card.splitlines() if line.startswith("| `")]
        assert "| `Shout` | 2 |" in rows
        assert "| `Calm` | 0 |" in rows
        assert "| `Whisper` | 1 |" in rows

    def test_it_keeps_the_style_table_in_the_manifests_own_order(self) -> None:
        """並べ替えを挟まない — manifest の並びを変えると表の並びも変わる。"""

        def style_rows(manifest: dict[str, Any]) -> list[str]:
            card = render_sbv2_model_card(manifest)
            head, _, rest = card.partition("## Styles")
            assert head  # 節が消えたら以下の抽出が無意味になる
            return [line for line in rest.splitlines() if line.startswith("| `")]

        manifest = _sbv2_manifest()
        assert style_rows(manifest)[:3] == [
            "| `Shout` | 2 |",
            "| `Calm` | 0 |",
            "| `Whisper` | 1 |",
        ]
        manifest["pipelineConfig"]["styles"] = {"Calm": 0, "Whisper": 1, "Shout": 2}
        assert style_rows(manifest)[:3] == [
            "| `Calm` | 0 |",
            "| `Whisper` | 1 |",
            "| `Shout` | 2 |",
        ]

    def test_it_renames_the_styles_when_the_manifest_does(self) -> None:
        """名前も焼き込まない（別 ckpt の配布形なら別のスタイル名が出る）。"""
        manifest = _sbv2_manifest()
        manifest["pipelineConfig"]["styles"] = {"Angry": 0}
        manifest["pipelineConfig"]["defaults"]["style"] = "Angry"
        card = render_sbv2_model_card(manifest)
        assert "| `Angry` | 0 |" in card
        for gone in ("Shout", "Calm", "Whisper"):
            assert gone not in card, gone

    def test_it_lists_every_speaker_with_its_row_id(self, sbv2_card: str) -> None:
        _, _, rest = sbv2_card.partition("## Speakers")
        rows = [line for line in rest.splitlines() if line.startswith("| `")]
        assert rows == ["| `ZZ9` | 0 |", "| `ZZ8` | 1 |"]

    def test_it_names_the_tables_the_ids_index(self, sbv2_card: str) -> None:
        """行番号の解決先も manifest のファイル path から引く（綴りを写経しない）。"""
        assert "`styles/fake_styles.safetensors`" in sbv2_card
        assert "`speakers/fake_spk.safetensors`" in sbv2_card

    def test_it_marks_exactly_the_default_preset_beside_its_name(self, sbv2_card: str) -> None:
        _, _, rest = sbv2_card.partition("## Presets")
        rows = [line for line in rest.splitlines() if line.startswith("| `")]
        default = [line for line in rows if "(default)" in line]
        assert len(default) == 1
        assert default[0].startswith("| `w8a8` (default) |")
        assert "| `f16` | `front` = `f16` / `voice` = `f16` | — |" in rows

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

    def test_it_does_not_invent_knobs_the_manifest_left_out(self) -> None:
        """既定の一覧は manifest の `defaults` そのもの（キーの写しを持たない）。"""
        manifest = _sbv2_manifest()
        manifest["pipelineConfig"]["defaults"] = {"speaker": "ZZ9", "style": "Calm"}
        card = render_sbv2_model_card(manifest)
        assert "- **speaker**: `ZZ9`" in card
        assert "lengthScale" not in card

    def test_the_usage_snippet_follows_the_manifest(self, sbv2_card: str) -> None:
        """スニペットの preset とスタイルも manifest 由来（存在しない綴りを勧めない）。"""
        assert "// The preset defaults to w8a8." in sbv2_card
        assert '  style: "Whisper",' in sbv2_card


class TestSbv2Determinism:
    def test_it_renders_the_same_bytes_for_the_same_manifest(self) -> None:
        first = render_sbv2_model_card(_sbv2_manifest())
        second = render_sbv2_model_card(json.loads(json.dumps(_sbv2_manifest())))
        assert first.encode("utf-8") == second.encode("utf-8")

    def test_it_does_not_mutate_the_manifest(self) -> None:
        manifest = _sbv2_manifest()
        before = copy.deepcopy(manifest)
        render_sbv2_model_card(manifest)
        assert manifest == before


class TestSbv2PipelineGate:
    def test_it_refuses_a_pipeline_it_does_not_describe(self) -> None:
        manifest = _sbv2_manifest()
        manifest["pipeline"] = SUPPORTED_PIPELINE
        with pytest.raises(ValueError, match=SUPPORTED_PIPELINE):
            render_sbv2_model_card(manifest)

    def test_the_two_templates_do_not_answer_for_each_other(self) -> None:
        """テンプレートは pipeline 固有 — 取り違えると「表は合うが説明が別モデル」になる。"""
        with pytest.raises(ValueError, match=SBV2_SUPPORTED_PIPELINE):
            render_model_card(_sbv2_manifest())
        with pytest.raises(ValueError, match=SUPPORTED_PIPELINE):
            render_sbv2_model_card(_manifest())
