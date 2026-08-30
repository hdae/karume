"""SBV2 配布形のモデルカード描画（`sbv2.card`）。

実物の配布形の `karume.json` は使わない — 偽 manifest の値がそのまま本文に出ること
（＝手書きの数値が 1 つも混ざっていないこと）を見るのがここの仕事なので、実物と**違う**値で
組んだほうが検出力が高い。スタイル表・話者表も同じ理由で実重み FN4 と別の名前・別の並びに
してある（表を焼き込んでいれば落ちる）。

manifest v2（`karume/2` — ADR 0041）以降、カードは 1 リポの複数モデルを説明する。ファミリーの
形（2 モデル・共有ファイルつき）を偽 manifest の既定にしてあるのは、単一モデルでしか通らない
描画を素通ししないため。

帰属はファミリー別プロファイル（`fn` / `jvnv`）。manifest からの導出を見る節は `fn` 1 本で
足りる（プロファイルは帰属節と frontmatter にしか掛からない）ので、プロファイル固有の事実と
取り違えの検出は末尾の `TestSbv2CardProfiles` にまとめてある。

NOTE: このテストだけ `siglip2.card` を import する（recipe 間で唯一のコード結合）—
{@link TestSbv2PipelineGate.test_the_two_templates_do_not_answer_for_each_other} は「2 つの
**実在テンプレート**が互いの manifest を拒む」ことの主張で、相手を合成に置き換えると主張が
割れる。SigLIP2 のテンプレートが core にあった頃から同じ被験体で、移設で指し先が変わっただけ。
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from sbv2.card import (
    SBV2_CARD_PROFILES,
    SBV2_FN_PROFILE,
    SBV2_JVNV_PROFILE,
    SBV2_SUPPORTED_PIPELINE,
    Sbv2CardProfile,
    render_sbv2_model_card,
)
from sbv2.distribution import SBV2_QUANT_ABBREVIATIONS
from siglip2.card import SIGLIP2_SUPPORTED_PIPELINE, render_siglip2_model_card

#: 使い方スニペットに綴られるリポ ID（組み立て先のディレクトリ名から dist が渡す）。
REPO = "hdae/fake-repo"


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


def _siglip2_manifest(model: str = "base") -> dict[str, Any]:
    """SigLIP2 の最小 manifest（テンプレート取り違えの門を両向きに見るための相手）。"""
    return {
        "format": "karume/4",
        "generator": "karume/9.9.9",
        "defaultModel": model,
        "models": {
            model: {
                "pipeline": SIGLIP2_SUPPORTED_PIPELINE,
                "weights": {
                    "vision": {"f32": {"shards": [_ref("v/model.f32.safetensors", 11, "c")]}}
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
    """ADR 0041 の形を保った最小の SBV2 manifest（値は実物と重ならない偽値）。

    `styles` は **ID の昇順に並べない** — カードが manifest の並びをそのまま出すこと
    （並べ替えを挟んでいないこと）を、そのまま観測できるようにするため。
    """
    text_encoder = _ref("shared/text_encoder/model.i8.safetensors", 555, "e")
    tokenizer = _ref("shared/tokenizer/fake-tokenizer.json", 11, "3")
    return {
        "format": "karume/4",
        "generator": "karume/9.9.9",
        "defaultModel": "ZA",
        "models": {
            "ZA": {
                "pipeline": SBV2_SUPPORTED_PIPELINE,
                "weights": {
                    "text_encoder": {"i8": {"shards": [text_encoder]}},
                    "front": {
                        "f16": {"shards": [_ref("ZA/front/model.f16.safetensors", 666, "f")]},
                        "i8": {"shards": [_ref("ZA/front/model.i8.safetensors", 777, "0")]},
                    },
                    "voice": {
                        "f16": {"shards": [_ref("ZA/voice/model.f16.safetensors", 888, "1")]},
                        "i8": {"shards": [_ref("ZA/voice/model.i8.safetensors", 999, "2")]},
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
                        "label": "偽のラベル",
                        "description": "偽の説明。",
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
                    "text_encoder": {"i8": {"shards": [text_encoder]}},
                    "front": {"i8": {"shards": [_ref("ZB/front/model.i8.safetensors", 100, "7")]}},
                    "voice": {"i8": {"shards": [_ref("ZB/voice/model.i8.safetensors", 200, "8")]}},
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
    """`fn` プロファイルで描いたカード（manifest 導出の観測はプロファイルに依らない）。"""
    return render_sbv2_model_card(_sbv2_manifest(), REPO, SBV2_FN_PROFILE, SBV2_QUANT_ABBREVIATIONS)


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
        assert "Sbv2Pipeline.fromPretrained({" in sbv2_card
        assert f'  repo: "{REPO}",' in sbv2_card
        assert '  // revision: "<full commit sha>",' in sbv2_card
        assert "@karume/models" in sbv2_card
        assert "using pipeline" in sbv2_card
        assert "encodeWav(audio.data, audio.sampleRate)" in sbv2_card
        assert "const audio = await pipeline.generate({" in sbv2_card


class TestSbv2QuantRounding:
    """quant 表に続く**丸め方**の備考（同じ `i4` でも GPTQ 校正付きか素の RTN かは表に出ない）。"""

    @staticmethod
    def _notes(card: str) -> str:
        _, _, rest = card.partition("## Model: ZA")
        _, _, notes = rest.partition("How the stored weights were rounded:")
        return notes.partition("### Styles")[0]

    def test_it_explains_only_the_quants_the_model_declares(self, sbv2_card: str) -> None:
        """配布形が持たない席は説明しない（`_sbv2_knob` と同じ規律）。"""
        notes = self._notes(sbv2_card)
        assert "- `i8` —" in notes
        assert "`i8+bert4`" not in notes
        assert "- `i4` —" not in notes

    def test_the_default_mark_follows_default_quant(self, sbv2_card: str) -> None:
        """既定マークは manifest 由来 — 備考に焼くと、既定が動いたとき表とだけ食い違う。"""
        assert "(default)" not in self._notes(sbv2_card)

        manifest = _sbv2_manifest()
        manifest["models"]["ZA"]["defaultQuant"] = "i8"
        moved = render_sbv2_model_card(manifest, REPO, SBV2_FN_PROFILE, SBV2_QUANT_ABBREVIATIONS)
        assert "- `i8` (default) —" in self._notes(moved)


class TestSbv2Derivation:
    """MUST: 数値・ファイル一覧・quant 表・スタイル表・話者表は manifest 由来。"""

    def test_the_file_table_lists_every_declared_path_once(self, sbv2_card: str) -> None:
        """表の行は宣言されたファイルと 1 対 1（宣言外の行も、重複した行も無い）。"""
        _, _, rest = sbv2_card.partition("## Model: ZA")
        files, _, _ = rest.partition("### Quants")
        model = _sbv2_manifest()["models"]["ZA"]
        declared = [
            ref["path"]
            for weights in model["weights"].values()
            for entry in weights.values()
            for ref in entry["shards"]
        ]
        declared += [ref["path"] for ref in model["assets"].values()]
        rows = [line for line in files.splitlines() if line.startswith("| `")]
        assert len(rows) == len(declared)
        for path in declared:
            assert files.count(f"`{path}`") == 1, path

    def test_it_takes_the_sizes_from_the_manifest(self, sbv2_card: str) -> None:
        manifest = _sbv2_manifest()
        manifest["models"]["ZA"]["weights"]["text_encoder"]["i8"]["shards"][0]["size"] = 12345
        moved = render_sbv2_model_card(manifest, REPO, SBV2_FN_PROFILE, SBV2_QUANT_ABBREVIATIONS)
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
            card = render_sbv2_model_card(manifest, REPO, SBV2_FN_PROFILE, SBV2_QUANT_ABBREVIATIONS)
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
        card = render_sbv2_model_card(manifest, REPO, SBV2_FN_PROFILE, SBV2_QUANT_ABBREVIATIONS)
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
        assert default[0].startswith("| `i8-a8` (default) |")
        assert (
            "| `f16` | — | `text_encoder` = `i8` / `front` = `f16` / `voice` = `f16` | — |" in rows
        )

    def test_it_prints_the_presentation_fields_the_manifest_carries(self, sbv2_card: str) -> None:
        """表示欄はカードにも同じ文字列で出る（ADR 0075 決定 5）。"""
        assert "**偽のラベル** — 偽の説明。" in sbv2_card

    def test_it_explains_the_abbreviation_the_seat_names_use(self, sbv2_card: str) -> None:
        """略称の対応は**必ず**出す（ADR 0074 決定 4）— `bert4` がどの部品の話か読めるように。"""
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
        """既定の一覧は manifest の `defaults` そのもの（キーの写しを持たない）。"""
        _, _, second = sbv2_card.partition("## Model: ZB")
        assert "- **speaker**: `YY1`" in second
        assert "noiseScaleW" not in second

    def test_the_usage_snippet_follows_the_manifest(self, sbv2_card: str) -> None:
        """スニペットの model / quant / スタイルも manifest 由来（存在しない綴りを勧めない）。

        既定は綴られたうえで**コメントアウト**（裁定 2026-08-12 の「コメントを外すだけ」形）
        で、選べる値は同じ行に manifest から列挙される（声 / スタイル / quant が増えれば
        列挙も追従する）。
        """
        assert '  // model: "ZA", // default — available: ZA / ZB' in sbv2_card
        assert '  // quant: "i8-a8", // default — available: f16 / i8 / i8-a8' in sbv2_card
        assert '  // style: "Whisper", // default — available: Calm / Shout / Whisper' in sbv2_card
        assert '  // speaker: "ZZ8", // default — available: ZZ8 / ZZ9' in sbv2_card

    def test_the_usage_snippet_carries_every_delivery_knob_the_manifest_defines(
        self, sbv2_card: str
    ) -> None:
        """`generate()` の残りの optional も既定つきでコメント併記する（値は manifest 由来）。"""
        for line in (
            "  // styleWeight: 0.75, //",
            "  // lengthScale: 1.5, //",
            "  // sdpRatio: 0.25, //",
            "  // noiseScale: 0.65, //",
            "  // noiseScaleW: 0.85, //",
        ):
            assert line in sbv2_card, line

    def test_the_usage_snippet_skips_a_knob_the_manifest_left_out(self) -> None:
        """`defaults` に無いノブは綴らない（`### Defaults` と同じ判断 — 値を捏造しない）。"""
        manifest = _sbv2_manifest()
        manifest["defaultModel"] = "ZB"  # noiseScaleW 等を持たないモデル
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
        """テンプレートは pipeline 固有 — 取り違えると「表は合うが説明が別モデル」になる。"""
        with pytest.raises(ValueError, match=SBV2_SUPPORTED_PIPELINE):
            render_siglip2_model_card(_sbv2_manifest(), REPO)
        with pytest.raises(ValueError, match=SIGLIP2_SUPPORTED_PIPELINE):
            render_sbv2_model_card(
                _siglip2_manifest(), REPO, SBV2_FN_PROFILE, SBV2_QUANT_ABBREVIATIONS
            )


# ---- 帰属プロファイル（fn / jvnv）--------------------------------------


def _profile_card(profile: Sbv2CardProfile) -> str:
    return render_sbv2_model_card(_sbv2_manifest(), REPO, profile, SBV2_QUANT_ABBREVIATIONS)


class TestSbv2CardProfiles:
    """帰属（出所・ライセンス・引用）だけがファミリーごとに違う席であることを見る節。

    誤帰属は表も使い方も正しいまま起きるので、配ってからでないと誰も気づけない — 「別の
    ファミリーの事実が 1 語も混ざっていない」ことを両向きに観測する。
    """

    def test_it_offers_exactly_the_families_it_can_attribute(self) -> None:
        assert sorted(SBV2_CARD_PROFILES) == ["fn", "jvnv"]
        assert SBV2_CARD_PROFILES["fn"] is SBV2_FN_PROFILE
        assert SBV2_CARD_PROFILES["jvnv"] is SBV2_JVNV_PROFILE

    def test_it_refuses_to_draw_a_card_without_naming_the_family(self) -> None:
        """既定のプロファイルを持たない — 省略で描けると誤帰属が沈黙で再発する。"""
        with pytest.raises(TypeError):
            render_sbv2_model_card(_sbv2_manifest(), REPO)  # type: ignore[call-arg]

    @pytest.mark.parametrize("name", sorted(SBV2_CARD_PROFILES))
    def test_every_profile_keeps_the_same_skeleton(self, name: str) -> None:
        """プロファイルが動かすのは frontmatter と帰属節だけ（節の並びも表も同じ形）。"""
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
        """SPDX 標準タグなので `license_name` / `license_link` は**並べない**。"""
        head = _profile_card(SBV2_JVNV_PROFILE).split("---")[1]
        assert "license: cc-by-sa-4.0" in head
        assert "license_name" not in head
        assert "license_link" not in head

    def test_the_jvnv_frontmatter_lists_both_redistributed_upstreams(self) -> None:
        """再配布しているのは声と text encoder の 2 本 — 関係はどちらも `quantized`。"""
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
        """BY = クレジット・出典・ライセンス URL・**加えた改変**の表示 / SA = 同一ライセンス。"""
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
            "**none of that code**",
        ):
            assert required in card, required

    def test_neither_family_carries_the_others_attribution(self) -> None:
        """片方の事実がもう片方へ 1 語も漏れない（誤帰属の観測はこの両向きが本体）。"""
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
    """カードが案内するロード入口 — `fromPretrained` の 1 本だけ。"""

    @pytest.mark.parametrize("name", sorted(SBV2_CARD_PROFILES))
    def test_it_does_not_advertise_the_local_asset_entry_point(self, name: str) -> None:
        """`fromAssets` は案内しない（2026-08-29 裁定）。

        分割配布形も読めるようになった（X2-101）が、あちらはバイト列を自分で持っている前提の
        ローカルデバッグ向けの面で、HF から使う読者の普通の入口は `fromPretrained`。両方を
        並べると「どちらを使うのか」を読者に判断させることになる。`fromPretrained` 側も併せて
        見るのは、Usage ごと消えても通る門にしないため。Usage 節はプロファイル共有なので、
        片方のプロファイルにだけ案内が生えることも含めて全プロファイルで見る。
        """
        card = _profile_card(SBV2_CARD_PROFILES[name])
        assert "fromAssets" not in card
        assert "Sbv2Pipeline.fromPretrained" in card
