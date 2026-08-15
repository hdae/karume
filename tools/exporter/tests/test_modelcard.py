"""モデルカードの描画（`karume.modelcard`）。

実物の配布形の `karume.json` は使わない — 偽 manifest の値がそのまま本文に出ること
（＝手書きの数値が 1 つも混ざっていないこと）を見るのがここの仕事なので、実物と**違う**値で
組んだほうが検出力が高い。

NOTE: Anima / SBV2 のカードテンプレートは wheel の外へ出た（ADR 0065 段 3+4）ので、その
依存ケースは `tools/export-recipes/<family>/tests/test_card.py` に居る。ここに残るのは
core の描画部品と、まだ core に居る family のテンプレート。
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from karume.modelcard import (
    BIREFNET_MODELS,
    BIREFNET_SUPPORTED_PIPELINE,
    BIREFNET_UPSTREAM,
    DEPTH_ANYTHING_LICENSE,
    DEPTH_ANYTHING_SUPPORTED_PIPELINE,
    DEPTH_ANYTHING_UPSTREAM,
    IRODORI_SUPPORTED_PIPELINE,
    SIGLIP2_SUPPORTED_PIPELINE,
    CardMetadata,
    _frontmatter,
    render_birefnet_model_card,
    render_depth_anything_model_card,
    render_siglip2_model_card,
)

#: 使い方スニペットに綴られるリポ ID（組み立て先のディレクトリ名から `karume.dist` が渡す）。
REPO = "hdae/fake-repo"


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


# ---- ① 共有部（frontmatter の任意席）------------------------------------------
#
# core に残る family はどれも `base_model_relation` / `license_name` / `license_link` を
# 持たない（格納形を変えない配布形 + SPDX 標準タグ）ので、**書いたときに出る**側の枝は
# family 経由では 1 つも踏まれなくなった（ADR 0065 段 3+4 で Anima / SBV2 が wheel の外へ）。
# 席そのものは core の責務なので、ここで直接見る。


class TestFrontmatterOptionalFields:
    """任意 3 席は「値があるときだけ 1 行出る」— 空欄で並べると別の主張になる。

    `license_name` / `license_link` を空で並べたカードは「名前の無い独自ライセンス」に読め、
    `base_model_relation` を空で並べたカードは HF の推論を殺す（CardMetadata の doc）。
    """

    @staticmethod
    def _metadata(**overrides: Any) -> CardMetadata:
        return CardMetadata(
            pipeline_tag="text-to-speech",
            base_model=("owner/base",),
            license="other",
            tags=("webgpu",),
            **overrides,
        )

    def test_it_omits_every_optional_field_that_is_unset(self) -> None:
        lines = _frontmatter(self._metadata())
        for absent in ("base_model_relation", "license_name", "license_link"):
            assert not any(line.startswith(absent) for line in lines), absent

    def test_it_writes_each_optional_field_that_is_set(self) -> None:
        lines = _frontmatter(
            self._metadata(
                base_model_relation="quantized",
                license_name="owner-terms",
                license_link="https://example.invalid/terms",
            )
        )
        assert "base_model_relation: quantized" in lines
        assert "license_name: owner-terms" in lines
        assert "license_link: https://example.invalid/terms" in lines


# ---- SigLIP2（image-feature-extraction）--------------------------------------
#
# manifest からの導出（表・数・使い方）は `tests/test_dist.py` の `TestSiglip2ModelCard` が
# **組み立て 1 周ぶん**で見る。ここが持つのは modelcard 側にしか無い門 — テンプレートの
# pipeline 固有性と、帰属表に無いモデルを描かないこと。


def _siglip2_manifest(model: str = "base") -> dict[str, Any]:
    """SigLIP2 の最小 manifest（値は実物と重ならない偽値）。"""
    return {
        "format": "karume/2",
        "generator": "karume/9.9.9",
        "defaultModel": model,
        "models": {
            model: {
                "pipeline": SIGLIP2_SUPPORTED_PIPELINE,
                "weights": {"vision": {"f32": {"file": _ref("v/model.f32.safetensors", 11, "c")}}},
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


class TestSiglip2CardGate:
    def test_it_refuses_a_pipeline_it_does_not_describe(self) -> None:
        manifest = _siglip2_manifest()
        manifest["models"]["base"]["pipeline"] = IRODORI_SUPPORTED_PIPELINE
        with pytest.raises(ValueError, match=SIGLIP2_SUPPORTED_PIPELINE):
            render_siglip2_model_card(manifest, REPO)

    def test_it_refuses_a_model_it_cannot_attribute(self) -> None:
        """帰属表に無いモデル名で描くと、`base_model` が 1 つ足りないカード（= 出所を名乗って
        いない再配布）が黙って出る。既知の一覧を添えて落とす。
        """
        with pytest.raises(ValueError, match="large"):
            render_siglip2_model_card(_siglip2_manifest("large"), REPO)

    def test_it_renders_the_same_bytes_for_the_same_manifest(self) -> None:
        manifest = _siglip2_manifest()
        before = copy.deepcopy(manifest)
        assert render_siglip2_model_card(manifest, REPO) == render_siglip2_model_card(
            manifest, REPO
        )
        assert manifest == before


# ---- BiRefNet 系（image-segmentation）----------------------------------------
#
# manifest からの導出（表・数・使い方）は `tests/test_dist.py` の `TestBirefnetModelCard` が
# **組み立て 1 周ぶん**で見る。ここが持つのは modelcard 側にしか無い門 — テンプレートの
# pipeline 固有性と、帰属表に無いモデルを描かないこと。


def _birefnet_manifest(model: str = "hr") -> dict[str, Any]:
    """BiRefNet 系の最小 manifest（値は実物と重ならない偽値）。"""
    return {
        "format": "karume/2",
        "generator": "karume/9.9.9",
        "defaultModel": model,
        "models": {
            model: {
                "pipeline": BIREFNET_SUPPORTED_PIPELINE,
                "weights": {"matte": {"f32": {"file": _ref("m/model.f32.safetensors", 13, "d")}}},
                "assets": {},
                "quants": {"f32": {"weights": {"matte": "f32"}, "session": {}}},
                "defaultQuant": "f32",
                "pipelineConfig": {
                    "imageWidth": 64,
                    "imageHeight": 96,
                    "imageMean": [0.1, 0.2, 0.3],
                    "imageStd": [0.4, 0.5, 0.6],
                    "interpolation": "bilinear",
                },
            }
        },
    }


class TestBirefnetCardGate:
    def test_it_refuses_a_pipeline_it_does_not_describe(self) -> None:
        manifest = _birefnet_manifest()
        manifest["models"]["hr"]["pipeline"] = IRODORI_SUPPORTED_PIPELINE
        with pytest.raises(ValueError, match=BIREFNET_SUPPORTED_PIPELINE):
            render_birefnet_model_card(manifest, REPO)

    def test_it_refuses_a_model_it_cannot_attribute(self) -> None:
        """帰属表に無いモデル名で描くと、`base_model` が 1 つ足りないカード（= 出所を名乗って
        いない再配布）が黙って出る。既知の一覧を添えて落とす。
        """
        with pytest.raises(ValueError, match="tiny"):
            render_birefnet_model_card(_birefnet_manifest("tiny"), REPO)

    def test_it_renders_the_same_bytes_for_the_same_manifest(self) -> None:
        manifest = _birefnet_manifest()
        before = copy.deepcopy(manifest)
        assert render_birefnet_model_card(manifest, REPO) == render_birefnet_model_card(
            manifest, REPO
        )
        assert manifest == before

    def test_the_title_follows_the_model(self) -> None:
        """上流が名前で売っているモデルはその名前で呼ぶ（見出しは帰属表 1 つから来る）。"""
        assert "# Lucida (BiRefNet) — Karume" in render_birefnet_model_card(
            _birefnet_manifest("lucida"), REPO
        )
        assert "# BiRefNet HR — Karume" in render_birefnet_model_card(_birefnet_manifest(), REPO)

    def test_the_upstream_table_is_the_only_source_of_the_repository_ids(self) -> None:
        """`BIREFNET_UPSTREAM` は帰属表からの導出（2 表にすると片方だけ動ける）。"""
        assert {name: entry.repo for name, entry in BIREFNET_MODELS.items()} == BIREFNET_UPSTREAM


# ---- Depth Anything V2（depth-estimation）------------------------------------
#
# manifest からの導出（表・数・使い方）は `tests/test_dist.py` の `TestDepthAnythingModelCard`
# が**組み立て 1 周ぶん**で見る。ここが持つのは modelcard 側にしか無い門 — テンプレートの
# pipeline 固有性と、帰属表に無いサイズを描かないこと。


def _depth_anything_manifest(model: str = "small") -> dict[str, Any]:
    """Depth Anything V2 の最小 manifest（値は実物と重ならない偽値）。"""
    return {
        "format": "karume/2",
        "generator": "karume/9.9.9",
        "defaultModel": model,
        "models": {
            model: {
                "pipeline": DEPTH_ANYTHING_SUPPORTED_PIPELINE,
                "weights": {"depth": {"f32": {"file": _ref("d/model.f32.safetensors", 17, "e")}}},
                "assets": {},
                "quants": {"f32": {"weights": {"depth": "f32"}, "session": {}}},
                "defaultQuant": "f32",
                "pipelineConfig": {
                    "imageWidth": 64,
                    "imageHeight": 96,
                    "imageMean": [0.1, 0.2, 0.3],
                    "imageStd": [0.4, 0.5, 0.6],
                    "interpolation": "bicubic",
                },
            }
        },
    }


class TestDepthAnythingCardGate:
    def test_it_refuses_a_pipeline_it_does_not_describe(self) -> None:
        manifest = _depth_anything_manifest()
        manifest["models"]["small"]["pipeline"] = IRODORI_SUPPORTED_PIPELINE
        with pytest.raises(ValueError, match=DEPTH_ANYTHING_SUPPORTED_PIPELINE):
            render_depth_anything_model_card(manifest, REPO)

    def test_it_refuses_a_size_it_cannot_attribute(self) -> None:
        """帰属表に無いサイズで描くと、CC BY-NC 4.0 の重みが Apache-2.0 の frontmatter を
        まとったカードになる（`DEPTH_ANYTHING_UPSTREAM` の MUST）。既知の一覧を添えて落とす。
        """
        for size in ("base", "large"):
            with pytest.raises(ValueError, match=size):
                render_depth_anything_model_card(_depth_anything_manifest(size), REPO)

    def test_it_renders_the_same_bytes_for_the_same_manifest(self) -> None:
        manifest = _depth_anything_manifest()
        before = copy.deepcopy(manifest)
        assert render_depth_anything_model_card(manifest, REPO) == render_depth_anything_model_card(
            manifest, REPO
        )
        assert manifest == before

    def test_the_attribution_table_only_holds_apache_licensed_sizes(self) -> None:
        """配れるサイズの表 = ライセンスの門（`karume.dist` の Depth Anything 節の MUST）。

        表を広げるときは `DEPTH_ANYTHING_LICENSE`（今は 1 値）をモデル単位へ割る改修と
        セットなので、ここが「広がったこと」を検出する席になる。
        """
        assert sorted(DEPTH_ANYTHING_UPSTREAM) == ["small"]
        assert DEPTH_ANYTHING_LICENSE == "apache-2.0"

    def test_the_title_comes_from_the_upstream_checkpoint_name(self) -> None:
        """見出しのサイズは帰属表 1 本から導く（2 表にすると別サイズとして売れてしまう）。"""
        card = render_depth_anything_model_card(_depth_anything_manifest(), REPO)
        assert "# Depth Anything V2 Small — Karume" in card
