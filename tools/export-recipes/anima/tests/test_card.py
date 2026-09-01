"""Anima のモデルカード描画（`anima.card`）— 公式リポと追加学習リポの 2 枚。

実物の `models/karume-anima/karume.json` は使わない — 偽 manifest の値がそのまま本文に出ること
（＝手書きの数値が 1 つも混ざっていないこと）を見るのがここの仕事なので、実物と**違う**値で
組んだほうが検出力が高い。数値・path・表示欄は実物と重ならない偽値のままで、**モデル名だけ**が
実物の綴り（{@link UPSTREAM_MODELS} のキー）— 出所節が manifest のモデル名から帰属を引くので、
架空の名前ではどちらのカードも 1 枚も描けない。

manifest v2（`karume/2` — ADR 0041）以降、カードは 1 リポの複数モデルを説明する。ファミリーの
形（2 モデル・共有ファイルつき）を偽 manifest の既定にしてあるのは、単一モデルでしか通らない
描画を素通ししないため。

カードが 2 枚になったのは 2026-09-01 の再構造（ADR 0087）— 公式リポ（CircleStone の 3 変種
同居・既定 = Turbo）と追加学習リポ（第三者 fine-tune）でライセンス告知も出所節の導入も違う。
Turbo が公式 checkpoint になったので **LoRA 焼き込みの帰属節そのものが消えた**（旧 turbo 専用
カードとその帰属門はこの波で削除）。
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from anima.card import (
    ANIMA_EXTRA_TITLE,
    ANIMA_OFFICIAL_TITLE,
    ATTRIBUTION_NOTICE,
    EXTRA_ORIGINS_INTRO,
    OFFICIAL_ORIGINS_INTRO,
    SUPPORTED_PIPELINE,
    UPSTREAM_MODELS,
    render_base_card,
    render_extra_card,
)
from anima.distribution import ANIMA_QUANT_ABBREVIATIONS

#: 使い方スニペットに綴られるリポ ID（組み立て先のディレクトリ名から dist が渡す）。
REPO = "hdae/fake-repo"

#: 公式リポのフィクスチャが並べるモデル（既定 = Turbo = guidance 1 の席）。
OFFICIAL_DEFAULT = "anima-turbo-v1.1"
OFFICIAL_SECOND = "anima-aesthetic-v1.1"

#: 追加学習リポのフィクスチャが並べるモデル（2 本目は `allowDifferentLicense` が false の出所 —
#: 再ライセンス禁止の注意書きが出る側）。
EXTRA_DEFAULT = "anima-wai-v1.0"
EXTRA_SECOND = "anima-copycat-20260610"


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


def _two_model_manifest(default_name: str, second_name: str) -> dict[str, Any]:
    """ADR 0041 の形を保った最小の manifest（値は実物と重ならない偽値）。

    2 モデルで、`shared/` の text_encoder を共有する形にしてある。既定モデルは guidance 1
    （= 負プロンプトが効かない席）、2 本目は CFG 側なので、1 枚のカードで両方の分岐を通る。
    """
    shared_encoder = _ref("shared/text_encoder/model.safetensors", 111, "a")
    rope = _ref(f"{default_name}/transformer/rope_base.safetensors", 333, "c")
    return {
        "format": "karume/4",
        "generator": "karume/9.9.9",
        "defaultModel": default_name,
        "models": {
            default_name: {
                "pipeline": SUPPORTED_PIPELINE,
                "weights": {
                    "text_encoder": {"f16": {"shards": [shared_encoder]}},
                    "transformer": {
                        "f16": {
                            "shards": [
                                _ref(f"{default_name}/transformer/model.f16.safetensors", 222, "b")
                            ],
                            "extras": {"rope_base": rope},
                        },
                        "i8": {
                            "shards": [
                                _ref(f"{default_name}/transformer/model.i8.safetensors", 444, "d")
                            ],
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
                    "f16+dit8-a8": {
                        "weights": {"text_encoder": "f16", "transformer": "i8"},
                        "session": {"linearCompute": "a8"},
                        "label": "偽のラベル",
                        "description": "偽の説明。",
                    },
                    "f16-c16": {
                        "weights": {"text_encoder": "f16", "transformer": "f16"},
                        "session": {"linearCompute": "f16"},
                        "gpuFeatures": {"shaderF16": True},
                    },
                },
                "defaultQuant": "f16+dit8-a8",
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
            second_name: {
                "pipeline": SUPPORTED_PIPELINE,
                "weights": {
                    "text_encoder": {"f16": {"shards": [shared_encoder]}},
                    "transformer": {
                        "i8": {
                            "shards": [
                                _ref(f"{second_name}/transformer/model.i8.safetensors", 666, "f")
                            ]
                        }
                    },
                },
                "assets": {"tokenizer": _ref("shared/tokenizer/qwen2.json", 555, "e")},
                "quants": {
                    "f16+dit8": {
                        "weights": {"text_encoder": "f16", "transformer": "i8"},
                        "session": {},
                    }
                },
                "defaultQuant": "f16+dit8",
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


def _official_manifest() -> dict[str, Any]:
    """公式リポ（karume-anima）の形 — 既定 = Turbo（guidance 1）+ 公式 Aesthetic。"""
    return _two_model_manifest(OFFICIAL_DEFAULT, OFFICIAL_SECOND)


def _extra_manifest() -> dict[str, Any]:
    """追加学習リポ（karume-anima-extra）の形 — 既定も CFG 側（多 step + guidance）。"""
    manifest = _two_model_manifest(EXTRA_DEFAULT, EXTRA_SECOND)
    defaults = manifest["models"][EXTRA_DEFAULT]["pipelineConfig"]["defaults"]
    defaults["guidanceScale"] = 4
    defaults["steps"] = 28
    return manifest


def _official(manifest: dict[str, Any]) -> str:
    return render_base_card(manifest, REPO, ANIMA_QUANT_ABBREVIATIONS)


def _extra(manifest: dict[str, Any]) -> str:
    return render_extra_card(manifest, REPO, ANIMA_QUANT_ABBREVIATIONS)


@pytest.fixture
def card() -> str:
    return _official(_official_manifest())


@pytest.fixture
def extra_card() -> str:
    return _extra(_extra_manifest())


def _sharded_manifest() -> dict[str, Any]:
    """1 コンポーネントが複数ファイルへ割れた配布形（1GiB 超の分割 — ADR 0071 / 0070 追記）。

    実配布の anima はこの形（transformer が 2〜4 shard）。単一 shard のフィクスチャしか無いと、
    「コンテナ = 1 個の safetensors ファイル」という散文が事実と食い違ったまま素通りする
    （実際に公開カードで起きた — X2-103）。
    """
    manifest = _official_manifest()
    manifest["models"][OFFICIAL_DEFAULT]["weights"]["transformer"]["f16"]["shards"] = [
        _ref(f"{OFFICIAL_DEFAULT}/transformer/model.f16-00001-of-00002.safetensors", 777, "1"),
        _ref(f"{OFFICIAL_DEFAULT}/transformer/model.f16-00002-of-00002.safetensors", 888, "2"),
    ]
    return manifest


@pytest.fixture
def sharded_card() -> str:
    return _official(_sharded_manifest())


class TestShardedDistribution:
    """分割された配布形（実配布の形）でカードが事実と食い違わないこと。"""

    def test_it_lists_every_shard_of_a_split_component(self, sharded_card: str) -> None:
        """ファイル表は shard を 1 本残らず並べる（`karume.modelcard` の導出 MUST の帰結）。"""
        for path in (
            f"{OFFICIAL_DEFAULT}/transformer/model.f16-00001-of-00002.safetensors",
            f"{OFFICIAL_DEFAULT}/transformer/model.f16-00002-of-00002.safetensors",
        ):
            assert sharded_card.count(f"`{path}`") == 1, path
        assert f"`{OFFICIAL_DEFAULT}/transformer/model.f16.safetensors`" not in sharded_card

    def test_it_never_calls_the_container_a_single_file(
        self, card: str, sharded_card: str, extra_card: str
    ) -> None:
        """散文は分割の有無に依らず 1 本（manifest 依存の出し分けはしない — X2-103 裁定 a）。

        `NOTICE.md` の改変列挙は Pipeline 構築時に固定で組まれ manifest を見られないので、
        カード側だけ出し分けると 2 つの文書が別のことを言う。したがって**どの manifest でも**
        「1 個の safetensors ファイル」とは書かない。
        """
        for text in (card, sharded_card, extra_card):
            flat = " ".join(text.split())
            assert "a single safetensors file" not in flat
            assert "split across numbered shards" in flat

    def test_it_does_not_advertise_the_local_asset_entry_point(
        self, card: str, sharded_card: str, extra_card: str
    ) -> None:
        """`fromAssets` は案内しない（2026-08-29 裁定）— HF から使う入口は `fromPretrained`。"""
        for text in (card, sharded_card, extra_card):
            assert "fromAssets" not in text


class TestOfficialCard:
    """公式リポのカード（`render_base_card`）— CircleStone の 3 変種が同居するリポの 1 枚。"""

    def test_it_titles_the_repository_as_the_official_one(self, card: str) -> None:
        assert f"# {ANIMA_OFFICIAL_TITLE}" in card
        assert f"# {ANIMA_EXTRA_TITLE}" not in card

    def test_it_introduces_every_model_as_an_official_release(self, card: str) -> None:
        """出所節の導入はリポごとに違う事実 — 取り違えると帰属の主張そのものが入れ替わる。"""
        for line in OFFICIAL_ORIGINS_INTRO:
            assert line in card
        assert EXTRA_ORIGINS_INTRO[0] not in card

    def test_it_never_mentions_a_baked_lora(self, card: str) -> None:
        """Turbo は公式 checkpoint になり、焼き込んだ LoRA が無くなった（2026-09-01）。

        帰属節が残っていると、焼いていない配布物が「この LoRA を焼いた」と名乗ることになる —
        値としては妥当な散文なので、読者が上流を辿って初めて食い違いに気づく。門を**全文**に
        掛けるのは、帰属節を消しても散文の 1 行が焼き込みを語り続けた実例があるため
        （`_usage` の step 説明が「the baked-in LoRA is distilled for…」のまま残っていた）。
        """
        assert "LoRA" not in card
        assert "## Baked-in LoRA" not in card
        assert "civitai.com/models/2560840" not in card
        assert "1b55e40bdb1d0e5a78cb498f245fccfdaae97823265db957d2aabdcf4cd3caf1" not in card

    def test_it_shows_the_attribution_notice_verbatim(self, card: str) -> None:
        """§3(b) の掲示要件は逐語で満たす（要約は掲示でない）。"""
        assert ATTRIBUTION_NOTICE in card

    def test_it_states_that_the_outputs_stay_the_users_own(self, card: str) -> None:
        """ライセンス v1.2 §2(e) — 非商用の縛りが掛かるのは重みで、生成物には掛からない。

        読者が最初に知りたい 1 点なので、License 節の中で名指しする（一次確認 2026-09-01）。
        """
        _, _, license_section = card.partition("## License")
        section, _, _ = license_section.partition("## Models")
        flat = " ".join(section.split())
        assert (
            "Outputs you generate are yours to use for any purpose, including commercially" in flat
        )
        assert "the non-commercial restriction applies to the model weights and derivatives" in flat

    def test_the_license_section_points_at_the_two_files_shipped_alongside(self, card: str) -> None:
        """§3(a) のライセンス文と §3(d) の Notice は同梱物 — カードはその在り処を言う。"""
        _, _, license_section = card.partition("## License")
        section, _, _ = license_section.partition("## Models")
        assert "`LICENSE.md`" in section
        assert "`NOTICE.md`" in section

    def test_the_license_section_denies_any_official_standing(self, card: str) -> None:
        """§3(d)(iii) — 公式製品・承認済みと誤認させない（折り位置に依らず**文**で見る）。"""
        _, _, license_section = card.partition("## License")
        section, _, _ = license_section.partition("## Models")
        assert (
            "not an official product of CircleStone Labs LLC, and it is not endorsed, approved or"
            " validated by CircleStone Labs LLC." in " ".join(section.split())
        )

    def test_it_stays_silent_about_third_party_permissions(self, card: str) -> None:
        """公式リポに掛かるのは CircleStone のライセンス 1 本だけ。

        第三者 fine-tune 向けの注意書き（出所ページの許諾・再ライセンス禁止）が常時出ると、
        読み手はこのリポにも別の条件が重なっていると読む。
        """
        assert "do not relicense it" not in card
        assert "community fine-tunes are redistributed" not in " ".join(card.split())

    def test_it_lists_the_origin_of_every_model_in_the_manifest(self, card: str) -> None:
        for name in (OFFICIAL_DEFAULT, OFFICIAL_SECOND):
            upstream = UPSTREAM_MODELS[name]
            assert f"### `{name}` — {upstream.title}" in card
            assert upstream.author in card
            assert upstream.source in card
            assert f"- **Converted from**: `{upstream.file}`" in card

    def test_it_lists_the_origins_in_manifest_order(self, card: str) -> None:
        """出所節の並びは manifest のまま — `models` 表・モデル別節と同じ順序にする。

        名前順に並べ替えると、既定モデルより後ろに来るはずのものが先頭へ出る（実際に
        `anima-copycat-…` が先頭に出て指摘された）。同じカードの中で 2 通りの順序が
        混在すると、読者は「この並びには意味がある」と読んでしまう。
        """
        order = list(_official_manifest()["models"])
        assert order != sorted(order), "名前順と一致する並びでは順序の主張を検査できない"
        positions = [card.index(f"### `{name}` — ") for name in order]
        assert positions == sorted(positions)

    def test_it_refuses_a_model_whose_origin_is_unknown(self) -> None:
        """出所表に無いモデルは**帰属を書けない** — 黙って省かず落とす。"""
        manifest = _official_manifest()
        manifest["models"]["anima-nope"] = manifest["models"][OFFICIAL_DEFAULT]

        with pytest.raises(ValueError, match=r"出所が card\.py に無い"):
            _official(manifest)

    def test_it_refuses_another_pipelines_manifest(self) -> None:
        manifest = _official_manifest()
        manifest["models"][OFFICIAL_DEFAULT]["pipeline"] = "sbv2/1"

        with pytest.raises(ValueError):
            _official(manifest)


class TestExtraCard:
    """追加学習リポのカード（`render_extra_card`）— 第三者 fine-tune だけが並ぶリポの 1 枚。"""

    def test_it_titles_the_repository_as_the_community_one(self, extra_card: str) -> None:
        assert f"# {ANIMA_EXTRA_TITLE}" in extra_card
        assert f"# {ANIMA_OFFICIAL_TITLE}" not in extra_card

    def test_it_introduces_every_model_as_a_community_fine_tune(self, extra_card: str) -> None:
        for line in EXTRA_ORIGINS_INTRO:
            assert line in extra_card
        assert OFFICIAL_ORIGINS_INTRO[0] not in extra_card

    def test_it_points_the_reader_at_the_official_repository(self, extra_card: str) -> None:
        """text stack は公式リポへの越境参照で持つ（`karume.json` の pinned 参照）。

        どこから重みが来るのかを書かないと、読者は「このリポだけで閉じている」と読む。
        """
        assert "hdae/karume-anima" in extra_card
        assert "The official models (base / Aesthetic /" in extra_card

    def test_it_never_mentions_a_baked_lora(self, extra_card: str) -> None:
        """LoRA を焼いていない配布物で LoRA を名乗ると、帰属そのものが嘘になる。"""
        assert "LoRA" not in extra_card

    def test_it_shows_the_attribution_notice_verbatim(self, extra_card: str) -> None:
        """§3(b) の掲示要件は公式リポ側と同じく逐語で満たす。"""
        assert ATTRIBUTION_NOTICE in extra_card

    def test_it_lists_the_permissions_of_every_source_page(self, extra_card: str) -> None:
        """出所ページの許諾欄は作者ごとに違う — **出所ごとに逐語で**載せる。"""
        for name in (EXTRA_DEFAULT, EXTRA_SECOND):
            for key, value in UPSTREAM_MODELS[name].permissions:
                assert f"- `{key}`: {value}" in extra_card

    def test_it_warns_when_a_source_forbids_relicensing(self, extra_card: str) -> None:
        """`allowDifferentLicense` が false の出所は「同じ条件で配る」ことを要求する。"""
        assert "allowDifferentLicense`: false" in extra_card
        assert "do not relicense it" in extra_card
        assert f"`{EXTRA_SECOND}`" in extra_card.partition("do not relicense it")[0]

    def test_it_stays_silent_about_relicensing_when_no_source_forbids_it(self) -> None:
        """条件の無いリポにだけ効く注意書きが常時出ると、読み手が条件を取り違える。"""
        manifest = _extra_manifest()
        del manifest["models"][EXTRA_SECOND]

        assert "do not relicense it" not in _extra(manifest)

    def test_it_never_lists_a_model_the_manifest_does_not_carry(self) -> None:
        """帰属を組み立ての引数から独立に持つと、入っていないモデルの出所が載る。"""
        manifest = _extra_manifest()
        del manifest["models"][EXTRA_SECOND]
        card = _extra(manifest)

        assert UPSTREAM_MODELS[EXTRA_SECOND].source not in card
        assert UPSTREAM_MODELS[EXTRA_DEFAULT].source in card

    def test_its_usage_snippet_keeps_the_cfg_knobs_as_defaults(self, extra_card: str) -> None:
        """CFG が既定で入っている配布物では、guidance / negative を「省略可の既定」として出す。"""
        assert "// guidanceScale: 4, // default" in extra_card
        assert "makes the negative prompt take effect" in extra_card

    def test_it_refuses_another_pipelines_manifest(self) -> None:
        manifest = _extra_manifest()
        manifest["models"][EXTRA_DEFAULT]["pipeline"] = "sbv2/1"

        with pytest.raises(ValueError):
            _extra(manifest)


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

    def test_both_repositories_declare_the_same_upstream_license(
        self, card: str, extra_card: str
    ) -> None:
        """どちらのリポも同じ base の Derivative — 上流ライセンスは 1 本しか掛からない。"""
        assert card.split("---")[1] == extra_card.split("---")[1]


class TestSections:
    def test_the_official_card_carries_every_section(self, card: str) -> None:
        headings = [line for line in card.splitlines() if line.startswith("## ")]
        assert headings == [
            "## What is this",
            "## Models and their origins",
            "## License",
            "## Models",
            "## Usage",
            f"## Model: {OFFICIAL_DEFAULT}",
            f"## Model: {OFFICIAL_SECOND}",
        ]

    def test_the_extra_card_carries_the_same_shape(self, extra_card: str) -> None:
        """2 枚のカードは**同じ骨格**（違うのは題・概要・出所節の導入だけ）。"""
        headings = [line for line in extra_card.splitlines() if line.startswith("## ")]
        assert headings == [
            "## What is this",
            "## Models and their origins",
            "## License",
            "## Models",
            "## Usage",
            f"## Model: {EXTRA_DEFAULT}",
            f"## Model: {EXTRA_SECOND}",
        ]

    def test_each_model_section_carries_its_own_tables(self, card: str) -> None:
        _, _, rest = card.partition(f"## Model: {OFFICIAL_DEFAULT}")
        first, _, second = rest.partition(f"## Model: {OFFICIAL_SECOND}")
        for part in (first, second):
            assert [line for line in part.splitlines() if line.startswith("### ")] == [
                "### Files",
                "### Quants",
                "### Defaults",
            ]

    def test_it_shows_the_minimal_typescript_entry_point(self, card: str) -> None:
        assert "AnimaPipeline.fromPretrained({" in card
        assert f'  repo: "{REPO}",' in card
        assert "@karume/models" in card
        assert "using pipeline" in card

    def test_the_usage_snippet_offers_the_revision_pin(self, card: str) -> None:
        """source は object ref 形 — revision を書く席が無いと読み手は暗黙に `main` 追従になる。"""
        assert '  // revision: "<full commit sha>",' in card
        assert "Pin a commit for reproducible builds" in card

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
            f"| `{OFFICIAL_DEFAULT}` (default) | `anima/1` |"
            " `f16` / `f16+dit8-a8` / `f16-c16` | `f16+dit8-a8` |",
            f"| `{OFFICIAL_SECOND}` | `anima/1` | `f16+dit8` | `f16+dit8` |",
        ]

    def test_the_usage_snippet_names_the_default_model_and_its_quant(self, card: str) -> None:
        """既定は綴られたうえで**コメントアウト**（裁定 2026-08-12 の「コメントを外すだけ」形）
        で、選べる値は同じ行に manifest から列挙される（モデル / quant が増えれば追従する）。
        """
        available = " / ".join(sorted((OFFICIAL_DEFAULT, OFFICIAL_SECOND)))
        assert f'  // model: "{OFFICIAL_DEFAULT}", // default — available: {available}' in card
        assert (
            '  // quant: "f16+dit8-a8", // default — available: f16 / f16+dit8-a8 / f16-c16'
        ) in card

    def test_it_follows_the_manifest_when_the_default_moves(self) -> None:
        manifest = _official_manifest()
        manifest["defaultModel"] = OFFICIAL_SECOND
        card = _official(manifest)
        available = " / ".join(sorted((OFFICIAL_DEFAULT, OFFICIAL_SECOND)))

        assert f'  // model: "{OFFICIAL_SECOND}", // default — available: {available}' in card
        assert '  // quant: "f16+dit8", // default — available: f16+dit8' in card
        assert f"| `{OFFICIAL_SECOND}` (default) |" in card


class TestDerivation:
    """MUST: 数値・ファイル一覧・quant 表は manifest 由来（手書きが混ざっていない）。"""

    def test_it_lists_every_declared_path_of_the_model_it_describes(self, card: str) -> None:
        _, _, rest = card.partition(f"## Model: {OFFICIAL_DEFAULT}")
        first, _, second = rest.partition(f"## Model: {OFFICIAL_SECOND}")
        for path in (
            "shared/text_encoder/model.safetensors",
            f"{OFFICIAL_DEFAULT}/transformer/model.f16.safetensors",
            f"{OFFICIAL_DEFAULT}/transformer/model.i8.safetensors",
            f"{OFFICIAL_DEFAULT}/transformer/rope_base.safetensors",
            "shared/tokenizer/qwen2.json",
        ):
            assert first.count(f"`{path}`") == 1, path
        assert second.count(f"`{OFFICIAL_SECOND}/transformer/model.i8.safetensors`") == 1
        assert f"{OFFICIAL_DEFAULT}/transformer" not in second

    def test_it_folds_a_shared_extra_into_one_row_naming_both_dtypes(self, card: str) -> None:
        row = next(line for line in card.splitlines() if "rope_base.safetensors`" in line)
        assert "`transformer.rope_base`" in row
        assert "f16 / i8" in row

    def test_an_asset_has_no_dtype_of_its_own(self, card: str) -> None:
        """assets は quant 選択に依存しない無条件ファイル（ADR 0041 §3）。"""
        row = next(line for line in card.splitlines() if "qwen2.json`" in line)
        assert row.startswith("| `tokenizer` | — |")

    def test_it_takes_the_sizes_from_the_manifest(self, card: str) -> None:
        manifest = _official_manifest()
        manifest["models"][OFFICIAL_DEFAULT]["weights"]["text_encoder"]["f16"]["shards"][0][
            "size"
        ] = 999
        moved = _official(manifest)
        assert "111 B" in card
        assert "111 B" not in moved
        assert "999 B" in moved

    def test_it_marks_exactly_the_default_quant_of_each_model(self, card: str) -> None:
        """印は**モデルごとに 1 つ**（既定は quant 表の持ち主が決める — ADR 0041 §2）。"""
        _, _, rest = card.partition(f"## Model: {OFFICIAL_DEFAULT}")
        first, _, second = rest.partition(f"## Model: {OFFICIAL_SECOND}")
        marked = {
            section.partition("### Quants")[2].partition("###")[0]: expected
            for section, expected in (
                (first, "| `f16+dit8-a8` (default) |"),
                (second, "| `f16+dit8` (default) |"),
            )
        }
        for quants, expected in marked.items():
            rows = [line for line in quants.splitlines() if line.startswith("| `")]
            default = [line for line in rows if "(default)" in line]
            assert len(default) == 1
            assert default[0].startswith(expected)
        # 表に「既定」列を持たない（印は名前の横だけ — 列が空欄で並ぶ形にしない）。
        assert "| Quant | What it is | Weights | Compute |" in card

    def test_it_carries_every_quant_with_its_session_knobs(self, card: str) -> None:
        assert "| `f16` | — | `text_encoder` = `f16` / `transformer` = `f16` | — |" in card
        assert "`linearCompute` = `a8`" in card
        assert "requires `shaderF16`" in card

    def test_it_prints_the_presentation_fields_the_manifest_carries(self, card: str) -> None:
        """表示欄はカードにも同じ文字列で出る（ADR 0075 決定 5 — 説明が 2 箇所で育たない）。"""
        assert "**偽のラベル** — 偽の説明。" in card

    def test_it_explains_the_abbreviation_the_seat_names_use(self, card: str) -> None:
        """略称の対応は**必ず**出す（ADR 0074 決定 4）— `dit8` がどの部品の話か読めるように。"""
        assert "In a quant name, `dit` is the `transformer` component." in card

    def test_it_takes_the_defaults_from_each_models_pipeline_config(self, card: str) -> None:
        assert "- **steps**: 7" in card
        assert "- **resolution**: 640 × 384" in card
        assert "ネガティブの偽値" in card
        assert "- **steps**: 4" in card
        assert "- **resolution**: 128 × 256" in card

    def test_it_only_warns_about_the_unused_negative_prompt_at_guidance_one(
        self, card: str
    ) -> None:
        """注意書きが出るのは guidance 1 の席だけ（偽 manifest では既定モデルの 1 本）。"""
        manifest = _official_manifest()
        manifest["models"][OFFICIAL_DEFAULT]["pipelineConfig"]["defaults"]["guidanceScale"] = 4

        assert "the negative prompt is not used" in card
        assert "the negative prompt is not used" not in _official(manifest)


class TestDeterminism:
    def test_it_renders_the_same_bytes_for_the_same_manifest(self) -> None:
        first = _official(_official_manifest())
        second = _official(json.loads(json.dumps(_official_manifest())))
        assert first.encode("utf-8") == second.encode("utf-8")

    def test_it_does_not_mutate_the_manifest(self) -> None:
        manifest = _official_manifest()
        before = copy.deepcopy(manifest)
        _official(manifest)
        assert manifest == before


class TestPipelineGate:
    def test_it_refuses_a_pipeline_it_does_not_describe(self) -> None:
        manifest = _official_manifest()
        manifest["models"][OFFICIAL_SECOND]["pipeline"] = "sbv2/1"
        with pytest.raises(ValueError, match="sbv2/1"):
            _official(manifest)
