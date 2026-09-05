"""BiRefNet 系配布形のモデルカード描画（`birefnet.card`）。

実物の配布形の `karume.json` は使わない — 偽 manifest の値がそのまま本文に出ること
（＝手書きの数値が 1 つも混ざっていないこと）を見るのがここの仕事なので、実物と**違う**値で
組んだほうが検出力が高い。

manifest からの導出（表・数・使い方）は `birefnet/tests/test_distribution.py` の
`TestBirefnetModelCard` が**組み立て 1 周ぶん**で見る。ここが持つのはカード側にしか無い門 —
テンプレートの pipeline 固有性と、帰属表に無いモデルを描かないこと、そして案内するロード入口。
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from birefnet.card import (
    BIREFNET_MAX_BINDING_TEXT,
    BIREFNET_MODELS,
    BIREFNET_RESOURCE_MEASUREMENT,
    BIREFNET_SUPPORTED_PIPELINE,
    BIREFNET_TOTAL_GPU_TEXT,
    BIREFNET_UPSTREAM,
    render_birefnet_model_card,
)

#: 使い方スニペットに綴られるリポ ID（組み立て先のディレクトリ名から dist が渡す）。
REPO = "hdae/fake-repo"

#: 別 family の pipeline 契約（**綴りをそのまま持つ** — recipe 間のコード結合を作らないため）。
#: 2 つの実在テンプレートが互いの manifest を拒むことは `sbv2/tests/test_card.py` が
#: 両向きで見る。ここが要るのは「自分の契約以外を拒む」という 1 方向だけ。
FOREIGN_PIPELINE = "siglip2/1"


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


def _birefnet_manifest(model: str = "hr") -> dict[str, Any]:
    """BiRefNet 系の最小 manifest（値は実物と重ならない偽値）。"""
    return {
        "format": "karume/3",
        "generator": "karume/9.9.9",
        "defaultModel": model,
        "models": {
            model: {
                "pipeline": BIREFNET_SUPPORTED_PIPELINE,
                "weights": {
                    "matte": {"f32": {"shards": [_ref("m/model.f32.safetensors", 13, "d")]}}
                },
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
        manifest["models"]["hr"]["pipeline"] = FOREIGN_PIPELINE
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


class TestBirefnetEntryPoint:
    """カードが案内するロード入口 — `fromPretrained` の 1 本だけ。"""

    def test_it_does_not_advertise_the_local_asset_entry_point(self) -> None:
        """`fromAssets` は案内しない（2026-08-29 裁定）。

        分割配布形も読めるようになった（X2-101）が、あちらはバイト列を自分で持っている前提の
        ローカルデバッグ向けの面で、HF から使う読者の普通の入口は `fromPretrained`。両方を
        並べると「どちらを使うのか」を読者に判断させることになる。`fromPretrained` 側も併せて
        見るのは、Usage ごと消えても通る門にしないため。
        """
        card = render_birefnet_model_card(_birefnet_manifest(), REPO)
        assert "fromAssets" not in card
        assert "BirefnetPipeline.fromPretrained" in card


class TestBirefnetResourceNote:
    """実行資源の注記（利用者が「渡す前に知りたい事実」— manifest に無いので定数）。"""

    def test_the_card_names_the_gpu_memory_it_needs(self) -> None:
        """MUST: 1 binding の大きさと総確保の両方を名乗る。

        WebGPU の `maxStorageBufferBindingSize` の仕様既定は 128MiB なので、1GiB 級の
        binding は「端末によっては要求自体が通らない」制約。カードが黙っていると、読み手は
        `requiredLimits` が空なこと（= 常駐分は既定内）を「既定スペックで動く」と読む。
        """
        card = render_birefnet_model_card(_birefnet_manifest(), REPO)

        assert BIREFNET_MAX_BINDING_TEXT in card
        assert BIREFNET_TOTAL_GPU_TEXT in card
        assert BIREFNET_RESOURCE_MEASUREMENT in card
        assert "maxStorageBufferBindingSize" in card
