"""リポの dist ドライバ（`tools/export-recipes/dist.py`）— 受理集合と置き場の既定の合成。

core の {@link karume.dist.PIPELINES} は wheel だけで組める分しか持たない（ADR 0065 決定 2）。
ここで固定するのは「recipe 側の family が 1 つ残らず表に載り、旧 UX の既定が保たれる」こと —
載せ忘れは `--pipeline <family>` が「そんな pipeline は無い」で落ちる形でしか表面化せず、
資産を作り終えた後に初めて気づく。

置き場の既定（`models/` / `outputs/series/`）もドライバの持ち物（ADR 0065 Consequences）—
core は綴りを持たないので、{@link dist.default_out_dir} の規則はここが見る。
"""

from __future__ import annotations

import pytest

import dist
from _shared.paths import DIST_ROOT, SERIES_ROOT
from anima.distribution import EXTRA_PIPELINE as ANIMA_EXTRA_PIPELINE
from anima.distribution import OFFICIAL_PIPELINE as ANIMA_OFFICIAL_PIPELINE
from birefnet.distribution import PIPELINE as BIREFNET_PIPELINE
from depth_anything.distribution import PIPELINE as DEPTH_ANYTHING_PIPELINE
from gemma4.distribution import PIPELINE as GEMMA4_PIPELINE
from irodori.distribution import PIPELINE as IRODORI_PIPELINE
from karume.dist import PIPELINES as CORE_PIPELINES
from karume.dist import DistError
from sbv2.distribution import PIPELINE as SBV2_PIPELINE
from siglip2.distribution import PIPELINE as SIGLIP2_PIPELINE
from vowel_detector.distribution import PIPELINE as VOWEL_DETECTOR_PIPELINE

#: 配布 recipe を持つ family の全量（名前 → その family が公開する `PIPELINE`）。
#: **ここが受理集合の期待値**で、`dist.PIPELINES` の載せ忘れも余剰も 1 つの表で検出する。
RECIPE_PIPELINES = {
    "anima": ANIMA_OFFICIAL_PIPELINE,
    "anima-extra": ANIMA_EXTRA_PIPELINE,
    "sbv2": SBV2_PIPELINE,
    "irodori": IRODORI_PIPELINE,
    "siglip2": SIGLIP2_PIPELINE,
    "birefnet": BIREFNET_PIPELINE,
    "depth-anything": DEPTH_ANYTHING_PIPELINE,
    "vowel-detector": VOWEL_DETECTOR_PIPELINE,
    "gemma4": GEMMA4_PIPELINE,
}


class TestRegistry:
    def test_it_carries_every_core_pipeline(self) -> None:
        for name, spec in CORE_PIPELINES.items():
            assert dist.PIPELINES[name] is spec

    def test_it_adds_every_recipe_pipeline(self) -> None:
        for name, spec in RECIPE_PIPELINES.items():
            assert dist.PIPELINES[name] is spec

    def test_the_table_holds_nothing_else(self) -> None:
        """余剰の席は「どの family の表でもない pipeline」— 名前だけで組める形にしない。"""
        assert sorted(dist.PIPELINES) == sorted({**RECIPE_PIPELINES, **CORE_PIPELINES})

    def test_the_core_table_is_empty_now_that_every_family_has_moved(self) -> None:
        """移行済み family が core 側に残っていれば、合成で 2 つの表が同じ名前を主張する。

        ADR 0065 段 3+4 の完了条件そのもの — core wheel に family 知識が 1 つも残っていない。
        """
        assert CORE_PIPELINES == {}

    def test_the_default_is_anima(self) -> None:
        """旧 `karume dist`（引数なし）の UX をドライバ側で維持する。"""
        assert dist.DEFAULT_PIPELINE == "anima"
        assert dist.DEFAULT_PIPELINE in dist.PIPELINES

    def test_the_two_anima_repositories_stay_separate_pipelines(self) -> None:
        """公式（`anima`）と追加学習（`anima-extra`）は**別の席**（ADR 0087 の分割軸）。

        `root_files` は Pipeline に固定で載る 1 組なので、1 つに畳むとどちらかのリポの
        改変告知が中身と食い違う — 散文としては妥当なままなので `verify_dist` も manifest
        検査も素通りし、配ってからでないと誰も気づけない。
        """
        official = dist.PIPELINES["anima"]
        extra = dist.PIPELINES["anima-extra"]

        assert official is not extra
        assert official.root_files["NOTICE.md"] != extra.root_files["NOTICE.md"]

    def test_every_pipeline_renders_its_own_model_card(self) -> None:
        """カードは pipeline ごとのテンプレート — 描き手が他 pipeline の manifest を拒む。"""
        for name, spec in dist.PIPELINES.items():
            manifest = {"models": {"m": {"pipeline": f"{name}/0"}}}
            for render_card in spec.card_profiles.values():
                with pytest.raises(ValueError):
                    render_card(manifest, "hdae/x")


class TestDefaultPlaces:
    """`--series` / `--out` の既定 — repo topology を知っているのはドライバだけ。"""

    def test_the_series_default_is_the_repository_series_root(self) -> None:
        """`--series` を省いた起動が読むのはリポの `outputs/series/`。"""
        assert dist.SERIES_ROOT is SERIES_ROOT

    def test_the_default_output_directory_follows_the_single_model(self) -> None:
        assert dist.default_out_dir(SBV2_PIPELINE, ["jvnv-F1"]).parent == DIST_ROOT
        assert dist.default_out_dir(SIGLIP2_PIPELINE, ["base"]).name == "karume-siglip2-base"

    def test_it_refuses_to_invent_a_family_repository_name(self) -> None:
        """ファミリーリポの名前（例 `karume-sbv2-jvnv`）はモデル名の並びからは決まらない。"""
        with pytest.raises(DistError, match="--out"):
            dist.default_out_dir(SBV2_PIPELINE, ["jvnv-F1", "jvnv-F2"])
