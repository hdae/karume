"""モデルカードの描画（`karume.modelcard`）。

実物の `models/anima-turbo/karume.json` は使わない — 偽 manifest の値がそのまま本文に出ること
（＝手書きの数値が 1 つも混ざっていないこと）を見るのがここの仕事なので、実物と**違う**値で
組んだほうが検出力が高い。SBV2 のスタイル表・話者表も同じ理由で実重み FN4 と別の名前・別の
並びにしてある（表を焼き込んでいれば落ちる）。

manifest v2（`karume/2` — ADR 0041）以降、カードは 1 リポの複数モデルを説明する。ファミリーの
形（2 モデル・共有ファイルつき）を偽 manifest の既定にしてあるのは、単一モデルでしか通らない
描画を素通ししないため。
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from karume.modelcard import (
    SBV2_SUPPORTED_PIPELINE,
    SUPPORTED_PIPELINE,
    render_model_card,
    render_sbv2_model_card,
)

#: 使い方スニペットに綴られるリポ ID（組み立て先のディレクトリ名から `karume.dist` が渡す）。
REPO = "hdae/fake-repo"


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


def _manifest() -> dict[str, Any]:
    """ADR 0041 の形を保った最小の manifest（値は実物と重ならない偽値）。

    2 モデルで、`shared/` の text_encoder を共有する形にしてある。
    """
    shared_encoder = _ref("shared/text_encoder/model.safetensors", 111, "a")
    rope = _ref("turbo/transformer/rope_base.safetensors", 333, "c")
    return {
        "format": "karume/2",
        "generator": "karume/9.9.9",
        "defaultModel": "turbo",
        "models": {
            "turbo": {
                "pipeline": SUPPORTED_PIPELINE,
                "weights": {
                    "text_encoder": {"f16": {"file": shared_encoder}},
                    "transformer": {
                        "f16": {
                            "file": _ref("turbo/transformer/model.f16.safetensors", 222, "b"),
                            "extras": {"rope_base": rope},
                        },
                        "i8": {
                            "file": _ref("turbo/transformer/model.i8.safetensors", 444, "d"),
                            "extras": {"rope_base": rope},
                        },
                    },
                },
                "assets": {"tokenizer": _ref("shared/tokenizer/qwen2.json", 555, "e")},
                "quants": {
                    "f16": {
                        "weights": {"text_encoder": "f16", "transformer": "f16"},
                        "session": {},
                    },
                    "w8a8": {
                        "weights": {"text_encoder": "f16", "transformer": "i8"},
                        "session": {"linearCompute": "i8a8"},
                    },
                    "f16-c16": {
                        "weights": {"text_encoder": "f16", "transformer": "f16"},
                        "session": {"linearCompute": "f16"},
                        "gpuFeatures": {"shaderF16": True},
                    },
                },
                "defaultQuant": "w8a8",
                "pipelineConfig": {
                    "scheduler": {"shift": 3, "numTrainTimesteps": 1000},
                    "defaults": {
                        "steps": 7,
                        "guidanceScale": 1,
                        "resolution": {"width": 640, "height": 384},
                        "negativePrompt": "ネガティブの偽値",
                    },
                },
            },
            "lite": {
                "pipeline": SUPPORTED_PIPELINE,
                "weights": {
                    "text_encoder": {"f16": {"file": shared_encoder}},
                    "transformer": {
                        "i8": {"file": _ref("lite/transformer/model.i8.safetensors", 666, "f")}
                    },
                },
                "assets": {"tokenizer": _ref("shared/tokenizer/qwen2.json", 555, "e")},
                "quants": {
                    "w8": {
                        "weights": {"text_encoder": "f16", "transformer": "i8"},
                        "session": {},
                    }
                },
                "defaultQuant": "w8",
                "pipelineConfig": {
                    "scheduler": {"shift": 3, "numTrainTimesteps": 1000},
                    "defaults": {
                        "steps": 4,
                        "guidanceScale": 2,
                        "resolution": {"width": 128, "height": 256},
                        "negativePrompt": "もう一つのネガティブ",
                    },
                },
            },
        },
    }


@pytest.fixture
def card() -> str:
    return render_model_card(_manifest(), REPO)


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
            "## Models",
            "## Usage",
            "## Model: turbo",
            "## Model: lite",
        ]

    def test_each_model_section_carries_its_own_tables(self, card: str) -> None:
        _, _, rest = card.partition("## Model: turbo")
        turbo, _, lite = rest.partition("## Model: lite")
        for part in (turbo, lite):
            assert [line for line in part.splitlines() if line.startswith("### ")] == [
                "### Files",
                "### Quants",
                "### Defaults",
            ]

    def test_it_names_the_merged_lora_with_its_source_and_hash(self, card: str) -> None:
        assert "Anima Turbo LoRA v0.2" in card
        assert "circlestone_labs" in card
        assert "https://civitai.com/models/2560840?modelVersionId=2979642" in card
        assert "1b55e40bdb1d0e5a78cb498f245fccfdaae97823265db957d2aabdcf4cd3caf1" in card

    def test_it_shows_the_minimal_typescript_entry_point(self, card: str) -> None:
        assert f'AnimaPipeline.fromPretrained("{REPO}", {{' in card
        assert "@karume/models" in card
        assert "using pipeline" in card


class TestModelSelection:
    """v2 で初めて機械可読になった軸（ADR 0041 §2）— 一覧・既定・使い方の 3 箇所に出る。"""

    def test_it_lists_every_model_with_its_quants(self, card: str) -> None:
        _, _, rest = card.partition("## Models")
        table, _, _ = rest.partition("## Usage")
        rows = [line for line in table.splitlines() if line.startswith("| `")]
        assert rows == [
            "| `turbo` (default) | `anima/1` | `f16` / `w8a8` / `f16-c16` | `w8a8` |",
            "| `lite` | `anima/1` | `w8` | `w8` |",
        ]

    def test_the_usage_snippet_names_the_default_model_and_its_quant(self, card: str) -> None:
        assert '  model: "turbo",' in card
        assert '  quant: "w8a8",' in card

    def test_it_follows_the_manifest_when_the_default_moves(self) -> None:
        manifest = _manifest()
        manifest["defaultModel"] = "lite"
        card = render_model_card(manifest, REPO)
        assert '  model: "lite",' in card
        assert '  quant: "w8",' in card
        assert "| `lite` (default) |" in card


class TestDerivation:
    """MUST: 数値・ファイル一覧・quant 表は manifest 由来（手書きが混ざっていない）。"""

    def test_it_lists_every_declared_path_of_the_model_it_describes(self, card: str) -> None:
        _, _, rest = card.partition("## Model: turbo")
        turbo, _, lite = rest.partition("## Model: lite")
        for path in (
            "shared/text_encoder/model.safetensors",
            "turbo/transformer/model.f16.safetensors",
            "turbo/transformer/model.i8.safetensors",
            "turbo/transformer/rope_base.safetensors",
            "shared/tokenizer/qwen2.json",
        ):
            assert turbo.count(f"`{path}`") == 1, path
        assert lite.count("`lite/transformer/model.i8.safetensors`") == 1
        assert "turbo/transformer" not in lite

    def test_it_folds_a_shared_extra_into_one_row_naming_both_dtypes(self, card: str) -> None:
        row = next(line for line in card.splitlines() if "rope_base.safetensors`" in line)
        assert "`transformer.rope_base`" in row
        assert "f16 / i8" in row

    def test_an_asset_has_no_dtype_of_its_own(self, card: str) -> None:
        """assets は quant 選択に依存しない無条件ファイル（ADR 0041 §3）。"""
        row = next(line for line in card.splitlines() if "qwen2.json`" in line)
        assert row.startswith("| `tokenizer` | — |")

    def test_it_takes_the_sizes_from_the_manifest(self, card: str) -> None:
        manifest = _manifest()
        manifest["models"]["turbo"]["weights"]["text_encoder"]["f16"]["file"]["size"] = 999
        moved = render_model_card(manifest, REPO)
        assert "111 B" in card
        assert "111 B" not in moved
        assert "999 B" in moved

    def test_it_marks_exactly_the_default_quant_of_each_model(self, card: str) -> None:
        """印は**モデルごとに 1 つ**（既定は quant 表の持ち主が決める — ADR 0041 §2）。"""
        _, _, rest = card.partition("## Model: turbo")
        turbo, _, lite = rest.partition("## Model: lite")
        marked = {
            section.partition("### Quants")[2].partition("###")[0]: expected
            for section, expected in ((turbo, "| `w8a8` (default) |"), (lite, "| `w8` (default) |"))
        }
        for quants, expected in marked.items():
            rows = [line for line in quants.splitlines() if line.startswith("| `")]
            default = [line for line in rows if "(default)" in line]
            assert len(default) == 1
            assert default[0].startswith(expected)
        # 表に「既定」列を持たない（印は名前の横だけ — 列が空欄で並ぶ形にしない）。
        assert "| Quant | Weights | Compute |" in card

    def test_it_carries_every_quant_with_its_session_knobs(self, card: str) -> None:
        assert "| `f16` | `text_encoder` = `f16` / `transformer` = `f16` | — |" in card
        assert "`linearCompute` = `i8a8`" in card
        assert "requires `shaderF16`" in card

    def test_it_takes_the_defaults_from_each_models_pipeline_config(self, card: str) -> None:
        assert "- **steps**: 7" in card
        assert "- **resolution**: 640 × 384" in card
        assert "ネガティブの偽値" in card
        assert "- **steps**: 4" in card
        assert "- **resolution**: 128 × 256" in card

    def test_it_only_warns_about_the_unused_negative_prompt_at_guidance_one(
        self, card: str
    ) -> None:
        assert card.count("the negative prompt is not used") == 1  # turbo だけ guidance 1
        manifest = _manifest()
        manifest["models"]["turbo"]["pipelineConfig"]["defaults"]["guidanceScale"] = 4
        assert "the negative prompt is not used" not in render_model_card(manifest, REPO)


class TestDeterminism:
    def test_it_renders_the_same_bytes_for_the_same_manifest(self) -> None:
        first = render_model_card(_manifest(), REPO)
        second = render_model_card(json.loads(json.dumps(_manifest())), REPO)
        assert first.encode("utf-8") == second.encode("utf-8")

    def test_it_does_not_mutate_the_manifest(self) -> None:
        manifest = _manifest()
        before = copy.deepcopy(manifest)
        render_model_card(manifest, REPO)
        assert manifest == before


class TestPipelineGate:
    def test_it_refuses_a_pipeline_it_does_not_describe(self) -> None:
        manifest = _manifest()
        manifest["models"]["lite"]["pipeline"] = "sbv2/1"
        with pytest.raises(ValueError, match="sbv2/1"):
            render_model_card(manifest, REPO)


# ---- SBV2（text-to-speech）---------------------------------------------------


def _sbv2_manifest() -> dict[str, Any]:
    """ADR 0041 の形を保った最小の SBV2 manifest（値は実物と重ならない偽値）。

    `styles` は **ID の昇順に並べない** — カードが manifest の並びをそのまま出すこと
    （並べ替えを挟んでいないこと）を、そのまま観測できるようにするため。
    """
    text_encoder = _ref("shared/text_encoder/model.i8.safetensors", 555, "e")
    tokenizer = _ref("shared/tokenizer/fake-tokenizer.json", 11, "3")
    return {
        "format": "karume/2",
        "generator": "karume/9.9.9",
        "defaultModel": "ZA",
        "models": {
            "ZA": {
                "pipeline": SBV2_SUPPORTED_PIPELINE,
                "weights": {
                    "text_encoder": {"i8": {"file": text_encoder}},
                    "front": {
                        "f16": {"file": _ref("ZA/front/model.f16.safetensors", 666, "f")},
                        "i8": {"file": _ref("ZA/front/model.i8.safetensors", 777, "0")},
                    },
                    "voice": {
                        "f16": {"file": _ref("ZA/voice/model.f16.safetensors", 888, "1")},
                        "i8": {"file": _ref("ZA/voice/model.i8.safetensors", 999, "2")},
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
                    "w8": {
                        "weights": {"text_encoder": "i8", "front": "i8", "voice": "i8"},
                        "session": {},
                    },
                    "w8a8": {
                        "weights": {"text_encoder": "i8", "front": "i8", "voice": "i8"},
                        "session": {"linearCompute": "i8a8"},
                    },
                },
                "defaultQuant": "w8a8",
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
                    "text_encoder": {"i8": {"file": text_encoder}},
                    "front": {"i8": {"file": _ref("ZB/front/model.i8.safetensors", 100, "7")}},
                    "voice": {"i8": {"file": _ref("ZB/voice/model.i8.safetensors", 200, "8")}},
                },
                "assets": {
                    "tokenizer": tokenizer,
                    "symbols": _ref("shared/text/symbols.json", 22, "4"),
                    "style_vectors": _ref("ZB/styles/fake_styles.safetensors", 55, "9"),
                    "speaker_embeddings": _ref("ZB/speakers/fake_spk.safetensors", 66, "a"),
                },
                "quants": {
                    "w8": {
                        "weights": {"text_encoder": "i8", "front": "i8", "voice": "i8"},
                        "session": {},
                    }
                },
                "defaultQuant": "w8",
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
    return render_sbv2_model_card(_sbv2_manifest(), REPO)


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
                "### Files",
                "### Quants",
                "### Styles",
                "### Speakers",
                "### Defaults",
            ]

    def test_it_attributes_both_upstream_models(self, sbv2_card: str) -> None:
        """帰属は manifest に無い事実 — 声の出所と、再配布する text encoder の両方。"""
        assert "rufflet17/voice_models" in sbv2_card
        assert "2.6.1-JP-Extra" in sbv2_card
        assert "ku-nlp/deberta-v2-large-japanese-char-wwm" in sbv2_card
        assert "cc-by-sa-4.0" in sbv2_card

    def test_it_shows_the_minimal_typescript_entry_point(self, sbv2_card: str) -> None:
        """実在する公開面だけを綴る（`packages/models/mod.ts` の 2 名 + 実シグネチャ）。"""
        assert f'Sbv2Pipeline.fromPretrained("{REPO}", {{' in sbv2_card
        assert "@karume/models" in sbv2_card
        assert "using pipeline" in sbv2_card
        assert "encodeWav(audio.data, audio.sampleRate)" in sbv2_card
        assert "const audio = await pipeline.generate({" in sbv2_card


class TestSbv2Derivation:
    """MUST: 数値・ファイル一覧・quant 表・スタイル表・話者表は manifest 由来。"""

    def test_the_file_table_lists_every_declared_path_once(self, sbv2_card: str) -> None:
        """表の行は宣言されたファイルと 1 対 1（宣言外の行も、重複した行も無い）。"""
        _, _, rest = sbv2_card.partition("## Model: ZA")
        files, _, _ = rest.partition("### Quants")
        model = _sbv2_manifest()["models"]["ZA"]
        declared = [
            entry["file"]["path"]
            for weights in model["weights"].values()
            for entry in weights.values()
        ]
        declared += [ref["path"] for ref in model["assets"].values()]
        rows = [line for line in files.splitlines() if line.startswith("| `")]
        assert len(rows) == len(declared)
        for path in declared:
            assert files.count(f"`{path}`") == 1, path

    def test_it_takes_the_sizes_from_the_manifest(self, sbv2_card: str) -> None:
        manifest = _sbv2_manifest()
        manifest["models"]["ZA"]["weights"]["text_encoder"]["i8"]["file"]["size"] = 12345
        moved = render_sbv2_model_card(manifest, REPO)
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
            card = render_sbv2_model_card(manifest, REPO)
            head, _, rest = card.partition("### Styles")
            assert head  # 節が消えたら以下の抽出が無意味になる
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
        """名前も焼き込まない（別 ckpt の配布形なら別のスタイル名が出る）。"""
        manifest = _sbv2_manifest()
        del manifest["models"]["ZB"]
        manifest["models"]["ZA"]["pipelineConfig"]["styles"] = {"Angry": 0}
        manifest["models"]["ZA"]["pipelineConfig"]["defaults"]["style"] = "Angry"
        card = render_sbv2_model_card(manifest, REPO)
        assert "| `Angry` | 0 |" in card
        for gone in ("Shout", "Calm", "Whisper"):
            assert gone not in card, gone

    def test_it_lists_every_speaker_with_its_row_id(self, sbv2_card: str) -> None:
        _, _, rest = sbv2_card.partition("### Speakers")
        rows = [line for line in rest.splitlines() if line.startswith("| `")]
        assert rows[:2] == ["| `ZZ9` | 0 |", "| `ZZ8` | 1 |"]

    def test_it_names_the_tables_the_ids_index(self, sbv2_card: str) -> None:
        """行番号の解決先も manifest のファイル path から引く（綴りを写経しない）。"""
        assert "`ZA/styles/fake_styles.safetensors`" in sbv2_card
        assert "`ZA/speakers/fake_spk.safetensors`" in sbv2_card
        assert "`ZB/styles/fake_styles.safetensors`" in sbv2_card

    def test_it_marks_exactly_the_default_quant_of_the_model(self, sbv2_card: str) -> None:
        _, _, rest = sbv2_card.partition("### Quants")
        rows = [line for line in rest.partition("### Styles")[0].splitlines()]
        default = [line for line in rows if "(default)" in line]
        assert len(default) == 1
        assert default[0].startswith("| `w8a8` (default) |")
        assert "| `f16` | `text_encoder` = `i8` / `front` = `f16` / `voice` = `f16` | — |" in rows

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
        """既定の一覧は manifest の `defaults` そのもの（キーの写しを持たない）。"""
        _, _, second = sbv2_card.partition("## Model: ZB")
        assert "- **speaker**: `YY1`" in second
        assert "noiseScaleW" not in second

    def test_the_usage_snippet_follows_the_manifest(self, sbv2_card: str) -> None:
        """スニペットの model / quant / スタイルも manifest 由来（存在しない綴りを勧めない）。"""
        assert '  model: "ZA",' in sbv2_card
        assert '  quant: "w8a8",' in sbv2_card
        assert '  style: "Whisper",' in sbv2_card


class TestSbv2Determinism:
    def test_it_renders_the_same_bytes_for_the_same_manifest(self) -> None:
        first = render_sbv2_model_card(_sbv2_manifest(), REPO)
        second = render_sbv2_model_card(json.loads(json.dumps(_sbv2_manifest())), REPO)
        assert first.encode("utf-8") == second.encode("utf-8")

    def test_it_does_not_mutate_the_manifest(self) -> None:
        manifest = _sbv2_manifest()
        before = copy.deepcopy(manifest)
        render_sbv2_model_card(manifest, REPO)
        assert manifest == before


class TestSbv2PipelineGate:
    def test_it_refuses_a_pipeline_it_does_not_describe(self) -> None:
        manifest = _sbv2_manifest()
        manifest["models"]["ZA"]["pipeline"] = SUPPORTED_PIPELINE
        with pytest.raises(ValueError, match=SUPPORTED_PIPELINE):
            render_sbv2_model_card(manifest, REPO)

    def test_the_two_templates_do_not_answer_for_each_other(self) -> None:
        """テンプレートは pipeline 固有 — 取り違えると「表は合うが説明が別モデル」になる。"""
        with pytest.raises(ValueError, match=SBV2_SUPPORTED_PIPELINE):
            render_model_card(_sbv2_manifest(), REPO)
        with pytest.raises(ValueError, match=SUPPORTED_PIPELINE):
            render_sbv2_model_card(_manifest(), REPO)
