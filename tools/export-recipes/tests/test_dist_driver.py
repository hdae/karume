"""リポの dist ドライバ（`tools/export-recipes/dist.py`）— 受理集合の合成。

core の {@link karume.dist.PIPELINES} は wheel だけで組める分しか持たない（ADR 0065 決定 2）。
ここで固定するのは「recipe 側の family が 1 つ残らず表に載り、旧 UX の既定が保たれる」こと —
載せ忘れは `--pipeline <family>` が「そんな pipeline は無い」で落ちる形でしか表面化せず、
資産を作り終えた後に初めて気づく。
"""

from __future__ import annotations

import pytest

import dist
from anima.distribution import PIPELINE as ANIMA_PIPELINE
from karume.dist import PIPELINES as CORE_PIPELINES
from sbv2.distribution import PIPELINE as SBV2_PIPELINE


class TestRegistry:
    def test_it_carries_every_core_pipeline(self) -> None:
        for name, spec in CORE_PIPELINES.items():
            assert dist.PIPELINES[name] is spec

    def test_it_adds_the_recipe_pipelines(self) -> None:
        assert dist.PIPELINES["anima"] is ANIMA_PIPELINE
        assert dist.PIPELINES["sbv2"] is SBV2_PIPELINE

    def test_the_core_table_no_longer_carries_a_migrated_family(self) -> None:
        """移行済み family が core 側に残っていれば、合成で 2 つの表が同じ名前を主張する。"""
        assert "anima" not in CORE_PIPELINES
        assert "sbv2" not in CORE_PIPELINES

    def test_the_default_is_anima(self) -> None:
        """旧 `karume dist`（引数なし）の UX をドライバ側で維持する。"""
        assert dist.DEFAULT_PIPELINE == "anima"
        assert dist.DEFAULT_PIPELINE in dist.PIPELINES

    def test_every_pipeline_renders_its_own_model_card(self) -> None:
        """カードは pipeline ごとのテンプレート — 描き手が他 pipeline の manifest を拒む。"""
        for name, spec in dist.PIPELINES.items():
            manifest = {"models": {"m": {"pipeline": f"{name}/0"}}}
            for render_card in spec.card_profiles.values():
                with pytest.raises(ValueError):
                    render_card(manifest, "hdae/x")
