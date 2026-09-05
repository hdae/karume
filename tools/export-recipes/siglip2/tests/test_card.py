"""SigLIP2 配布形のモデルカード描画（`siglip2.card`）。

実物の配布形の `karume.json` は使わない — 偽 manifest の値がそのまま本文に出ること
（＝手書きの数値が 1 つも混ざっていないこと）を見るのがここの仕事なので、実物と**違う**値で
組んだほうが検出力が高い。

manifest からの導出（表・数・使い方）は `siglip2/tests/test_distribution.py` の
`TestSiglip2ModelCard` が**組み立て 1 周ぶん**で見る。ここが持つのはカード側にしか無い門 —
テンプレートの pipeline 固有性と、帰属表に無いモデルを描かないこと、そして案内するロード入口。
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from siglip2.card import (
    SIGLIP2_MAP_HEAD_DIFF,
    SIGLIP2_MAP_HEAD_NORM,
    SIGLIP2_SUPPORTED_PIPELINE,
    render_siglip2_model_card,
)

#: 使い方スニペットに綴られるリポ ID（組み立て先のディレクトリ名から dist が渡す）。
REPO = "hdae/fake-repo"

#: 別 family の pipeline 契約（**綴りをそのまま持つ** — recipe 間のコード結合を作らないため）。
#: 2 つの実在テンプレートが互いの manifest を拒むことは `sbv2/tests/test_card.py` が
#: 両向きで見る。ここが要るのは「自分の契約以外を拒む」という 1 方向だけ。
FOREIGN_PIPELINE = "birefnet/1"


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


def _siglip2_manifest(model: str = "base") -> dict[str, Any]:
    """SigLIP2 の最小 manifest（値は実物と重ならない偽値）。"""
    return {
        "format": "karume/3",
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


class TestSiglip2CardGate:
    def test_it_refuses_a_pipeline_it_does_not_describe(self) -> None:
        manifest = _siglip2_manifest()
        manifest["models"]["base"]["pipeline"] = FOREIGN_PIPELINE
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


class TestSiglip2LegalPointers:
    """リポ直下の法的テキスト（ADR 0092 決定 7）へカード本文から辿れること。"""

    def test_it_points_at_the_bundled_license_and_notice(self) -> None:
        card = render_siglip2_model_card(_siglip2_manifest(), REPO)
        assert "a verbatim copy is in `LICENSE.md`" in card
        assert "also listed in `NOTICE.md`, per Apache 2.0 §4(b)" in card

    def test_it_reports_the_measured_map_head_difference(self) -> None:
        """帰属節が名乗る差は実測の綴り 1 つきり（正本は `siglip2.measurements`）。"""
        card = render_siglip2_model_card(_siglip2_manifest(), REPO)
        assert SIGLIP2_MAP_HEAD_DIFF in card

    def test_it_reports_the_scale_that_gives_the_difference_its_meaning(self) -> None:
        """差の大小はスケール（L2 ノルム）との比でしか読めない — その数も綴りは 1 つきり。

        カードと `NOTICE.md` が別々に直書きすると、片方だけ動いた日に「告知だけが古い数を
        主張する」形が黙って作れる（`siglip2.measurements` の MUST）。
        """
        card = render_siglip2_model_card(_siglip2_manifest(), REPO)
        assert f"L2 norm is {SIGLIP2_MAP_HEAD_NORM}" in card


class TestSiglip2EntryPoint:
    """カードが案内するロード入口 — `fromPretrained` の 1 本だけ。"""

    def test_it_does_not_advertise_the_local_asset_entry_point(self) -> None:
        """`fromAssets` は案内しない（2026-08-29 裁定）。

        分割配布形も読めるようになった（X2-101）が、あちらはバイト列を自分で持っている前提の
        ローカルデバッグ向けの面で、HF から使う読者の普通の入口は `fromPretrained`。両方を
        並べると「どちらを使うのか」を読者に判断させることになる。`fromPretrained` 側も併せて
        見るのは、Usage ごと消えても通る門にしないため。
        """
        card = render_siglip2_model_card(_siglip2_manifest(), REPO)
        assert "fromAssets" not in card
        assert "Siglip2Pipeline.fromPretrained" in card
