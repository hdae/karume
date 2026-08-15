"""Depth Anything V2 配布形のモデルカード描画（`depth_anything.card`）。

実物の配布形の `karume.json` は使わない — 偽 manifest の値がそのまま本文に出ること
（＝手書きの数値が 1 つも混ざっていないこと）を見るのがここの仕事なので、実物と**違う**値で
組んだほうが検出力が高い。

manifest からの導出（表・数・使い方）は `depth_anything/tests/test_distribution.py` の
`TestDepthAnythingModelCard` が**組み立て 1 周ぶん**で見る。ここが持つのはカード側にしか無い
門 — テンプレートの pipeline 固有性と、帰属表に無いサイズを描かないこと。
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from depth_anything.card import (
    DEPTH_ANYTHING_LICENSE,
    DEPTH_ANYTHING_SUPPORTED_PIPELINE,
    DEPTH_ANYTHING_UPSTREAM,
    render_depth_anything_model_card,
)

#: 使い方スニペットに綴られるリポ ID（組み立て先のディレクトリ名から dist が渡す）。
REPO = "hdae/fake-repo"

#: 別 family の pipeline 契約（**綴りをそのまま持つ** — recipe 間のコード結合を作らないため）。
#: 2 つの実在テンプレートが互いの manifest を拒むことは `sbv2/tests/test_card.py` が
#: 両向きで見る。ここが要るのは「自分の契約以外を拒む」という 1 方向だけ。
FOREIGN_PIPELINE = "siglip2/1"


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


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
        manifest["models"]["small"]["pipeline"] = FOREIGN_PIPELINE
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
        """配れるサイズの表 = ライセンスの門（`depth_anything.distribution` の冒頭の MUST）。

        表を広げるときは `DEPTH_ANYTHING_LICENSE`（今は 1 値）をモデル単位へ割る改修と
        セットなので、ここが「広がったこと」を検出する席になる。
        """
        assert sorted(DEPTH_ANYTHING_UPSTREAM) == ["small"]
        assert DEPTH_ANYTHING_LICENSE == "apache-2.0"

    def test_the_title_comes_from_the_upstream_checkpoint_name(self) -> None:
        """見出しのサイズは帰属表 1 本から導く（2 表にすると別サイズとして売れてしまう）。"""
        card = render_depth_anything_model_card(_depth_anything_manifest(), REPO)
        assert "# Depth Anything V2 Small — Karume" in card
