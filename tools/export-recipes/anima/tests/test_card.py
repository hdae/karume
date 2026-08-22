"""Anima のモデルカード描画（`anima.card`）。

実物の `models/anima-turbo/karume.json` は使わない — 偽 manifest の値がそのまま本文に出ること
（＝手書きの数値が 1 つも混ざっていないこと）を見るのがここの仕事なので、実物と**違う**値で
組んだほうが検出力が高い。

manifest v2（`karume/2` — ADR 0041）以降、カードは 1 リポの複数モデルを説明する。ファミリーの
形（2 モデル・共有ファイルつき）を偽 manifest の既定にしてあるのは、単一モデルでしか通らない
描画を素通ししないため。
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from anima.card import (
    ATTRIBUTION_NOTICE,
    LORA_NAME,
    LORA_SOURCE,
    SUPPORTED_PIPELINE,
    UPSTREAM_MODELS,
    render_base_card,
    render_model_card,
)

#: 使い方スニペットに綴られるリポ ID（組み立て先のディレクトリ名から dist が渡す）。
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
        "format": "karume/3",
        "generator": "karume/9.9.9",
        "defaultModel": "turbo",
        "models": {
            "turbo": {
                "pipeline": SUPPORTED_PIPELINE,
                "weights": {
                    "text_encoder": {"f16": {"shards": [shared_encoder]}},
                    "transformer": {
                        "f16": {
                            "shards": [_ref("turbo/transformer/model.f16.safetensors", 222, "b")],
                            "extras": {"rope_base": rope},
                        },
                        "i8": {
                            "shards": [_ref("turbo/transformer/model.i8.safetensors", 444, "d")],
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
                    "text_encoder": {"f16": {"shards": [shared_encoder]}},
                    "transformer": {
                        "i8": {"shards": [_ref("lite/transformer/model.i8.safetensors", 666, "f")]}
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


def _base_manifest() -> dict[str, Any]:
    """素の base 系リポの形（base 本体 + 第三者 fine-tune）— モデル名は出所表に在るものだけ。"""
    manifest = _manifest()
    models = manifest["models"]
    models["anima-v1.0"] = models.pop("turbo")
    models["anima-copycat-20260610"] = models.pop("lite")
    manifest["defaultModel"] = "anima-v1.0"
    models["anima-v1.0"]["pipelineConfig"]["defaults"]["guidanceScale"] = 4
    models["anima-v1.0"]["pipelineConfig"]["defaults"]["steps"] = 28
    return manifest


@pytest.fixture
def base_card() -> str:
    return render_base_card(_base_manifest(), REPO)


class TestBaseCard:
    """素の base 系カード — turbo 前提の記述が 1 つも残っていないこと + 出所の帰属。"""

    def test_it_never_mentions_the_baked_lora(self, base_card: str) -> None:
        """LoRA を焼いていない配布物で「Baked-in LoRA」を名乗ると帰属そのものが嘘になる。"""
        assert LORA_NAME not in base_card
        assert LORA_SOURCE not in base_card
        assert "Baked-in LoRA" not in base_card

    def test_it_shows_the_attribution_notice_verbatim(self, base_card: str) -> None:
        """§3(b) の掲示要件は turbo 側と同じく逐語で満たす。"""
        assert ATTRIBUTION_NOTICE in base_card

    def test_it_lists_the_origin_of_every_model_in_the_manifest(self, base_card: str) -> None:
        for name in ("anima-v1.0", "anima-copycat-20260610"):
            upstream = UPSTREAM_MODELS[name]
            assert f"### `{name}` — {upstream.title}" in base_card
            assert upstream.author in base_card
            assert upstream.source in base_card

    def test_it_lists_the_origins_in_manifest_order(self, base_card: str) -> None:
        """出所節の並びは manifest のまま — `models` 表・モデル別節と同じ順序にする。

        名前順に並べ替えると、既定モデルより先に第三者 fine-tune が来る（実際に
        `anima-copycat-…` が先頭に出て指摘された）。同じカードの中で 2 通りの順序が
        混在すると、読者は「この並びには意味がある」と読んでしまう。
        """
        order = [name for name in _base_manifest()["models"]]
        assert order != sorted(order), "名前順と一致する並びでは順序の主張を検査できない"
        positions = [base_card.index(f"### `{name}` — ") for name in order]
        assert positions == sorted(positions)

    def test_it_never_lists_a_model_the_manifest_does_not_carry(self) -> None:
        """帰属を組み立ての引数から独立に持つと、入っていないモデルの出所が載る。"""
        manifest = _base_manifest()
        del manifest["models"]["anima-copycat-20260610"]
        card = render_base_card(manifest, REPO)

        assert UPSTREAM_MODELS["anima-copycat-20260610"].source not in card
        assert UPSTREAM_MODELS["anima-v1.0"].source in card

    def test_it_warns_when_a_source_forbids_relicensing(self, base_card: str) -> None:
        """`allowDifferentLicense` が false の出所は「同じ条件で配る」ことを要求する。"""
        assert "allowDifferentLicense`: false" in base_card
        assert "do not relicense it" in base_card

    def test_it_stays_silent_about_relicensing_when_no_source_forbids_it(self) -> None:
        """条件の無いリポにだけ効く注意書きが常時出ると、読み手が条件を取り違える。"""
        manifest = _base_manifest()
        del manifest["models"]["anima-copycat-20260610"]

        assert "do not relicense it" not in render_base_card(manifest, REPO)

    def test_it_refuses_a_model_whose_origin_is_unknown(self) -> None:
        """出所表に無いモデルは**帰属を書けない** — 黙って省かず落とす。"""
        manifest = _base_manifest()
        manifest["models"]["anima-nope"] = manifest["models"]["anima-v1.0"]

        with pytest.raises(ValueError, match=r"出所が card\.py に無い"):
            render_base_card(manifest, REPO)

    def test_its_usage_snippet_keeps_the_cfg_knobs_as_defaults(self, base_card: str) -> None:
        """CFG が既定で入っている配布物では、guidance / negative を「省略可の既定」として出す。"""
        assert "// guidanceScale: 4, // default" in base_card
        assert "makes the negative prompt take effect" in base_card

    def test_it_refuses_another_pipelines_manifest(self) -> None:
        manifest = _base_manifest()
        manifest["models"]["anima-v1.0"]["pipeline"] = "sbv2/1"

        with pytest.raises(ValueError):
            render_base_card(manifest, REPO)


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
            "## License",
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

    def test_the_license_section_displays_the_attribution_notice_verbatim(self, card: str) -> None:
        """§3(b) は掲示を求める — HF で最初に読まれるカードに逐語で出す（要約は掲示でない）。"""
        assert ATTRIBUTION_NOTICE in card

    def test_the_license_section_points_at_the_two_files_shipped_alongside(self, card: str) -> None:
        """§3(a) のライセンス文と §3(d) の Notice は同梱物 — カードはその在り処を言う。"""
        _, _, license_section = card.partition("## License")
        section, _, _ = license_section.partition("## Models")
        assert "`LICENSE.md`" in section
        assert "`NOTICE.md`" in section

    def test_the_license_section_names_the_baked_in_lora_and_denies_endorsement(
        self, card: str
    ) -> None:
        """§3(d)(i) / (iii) — 何を焼いたかと、公式製品でないことを帰属の隣で言う。"""
        _, _, license_section = card.partition("## License")
        section, _, _ = license_section.partition("## Models")
        assert f"the official {LORA_NAME} ([source]({LORA_SOURCE}))" in section
        # 折り位置に依らず**文**で見る（法的文言の単位は文）。
        assert (
            "not an official product of CircleStone Labs LLC, and it is not endorsed, approved or"
            " validated by CircleStone Labs LLC." in " ".join(section.split())
        )

    def test_it_shows_the_minimal_typescript_entry_point(self, card: str) -> None:
        assert f'AnimaPipeline.fromPretrained("{REPO}", {{' in card
        assert "@karume/models" in card
        assert "using pipeline" in card

    def test_the_usage_snippet_carries_the_optional_knobs_commented_out(self, card: str) -> None:
        """裁定 2026-08-12 の形: 動く最小形 + `generate()` の optional をコメントで併記する
        （値は manifest 由来 — steps も解像度もこの偽 manifest の値がそのまま出る）。
        """
        assert "  // steps: 7, //" in card
        assert "  // resolution: { width: 640, height: 384 }, // default" in card
        # guidance と negativePrompt は**対**で意味を持つ（guidanceScale 1 では後者が拒まれる）。
        assert "  // guidanceScale: 5," in card
        assert '  // negativePrompt: "ネガティブの偽値",' in card


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
        """既定は綴られたうえで**コメントアウト**（裁定 2026-08-12 の「コメントを外すだけ」形）
        で、選べる値は同じ行に manifest から列挙される（モデル / quant が増えれば追従する）。
        """
        assert '  // model: "turbo", // default — available: lite / turbo' in card
        assert '  // quant: "w8a8", // default — available: f16 / f16-c16 / w8a8' in card

    def test_it_follows_the_manifest_when_the_default_moves(self) -> None:
        manifest = _manifest()
        manifest["defaultModel"] = "lite"
        card = render_model_card(manifest, REPO)
        assert '  // model: "lite", // default — available: lite / turbo' in card
        assert '  // quant: "w8", // default — available: w8' in card
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
        manifest["models"]["turbo"]["weights"]["text_encoder"]["f16"]["shards"][0]["size"] = 999
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
