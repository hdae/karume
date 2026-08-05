"""モデルカードの描画（`karume.modelcard`）。

実物の `models/anima-turbo/karume.json` は使わない — 偽 manifest の値がそのまま本文に出ること
（＝手書きの数値が 1 つも混ざっていないこと）を見るのがここの仕事なので、実物と**違う**値で
組んだほうが検出力が高い。
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from karume.modelcard import HF_REPO, SUPPORTED_PIPELINE, render_model_card


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
