"""BiRefNet 系配布形のモデルカード描画（`birefnet.card`）。

実物の配布形の `karume.json` は使わない — 偽 manifest の値がそのまま本文に出ること
（＝手書きの数値が 1 つも混ざっていないこと）を見るのがここの仕事なので、実物と**違う**値で
組んだほうが検出力が高い。**例外は解像度**（`imageWidth` / `imageHeight`）: 実行資源の実測表
（{@link BIREFNET_RESOURCES}）はこの数で引くので、実測していない解像度では描けない。

manifest からの導出（表・数・使い方）は `birefnet/tests/test_distribution.py` の
`TestBirefnetModelCard` / `TestBirefnetResolutionFamily` が**組み立て 1 周ぶん**で見る。ここが
持つのはカード側にしか無い門 — テンプレートの pipeline 固有性と、帰属表に無い checkpoint を
描かないこと、実測していない解像度を描かないこと、そして案内するロード入口。
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from birefnet.card import (
    BIREFNET_CHECKPOINTS,
    BIREFNET_RESOURCE_MEASUREMENT,
    BIREFNET_RESOURCES,
    BIREFNET_SUPPORTED_PIPELINE,
    BIREFNET_UPSTREAM,
    render_birefnet_model_card,
)

#: 使い方スニペットに綴られるリポ ID（組み立て先のディレクトリ名から dist が渡す）。
REPO = "hdae/fake-repo"

#: このリポが配る重み（pipeline 席が渡す軸 — manifest には無い）。
CHECKPOINT = "hr"

#: 既定のモデル名 = 既定の解像度。
MODEL = "1024"

#: 別 family の pipeline 契約（**綴りをそのまま持つ** — recipe 間のコード結合を作らないため）。
#: 2 つの実在テンプレートが互いの manifest を拒むことは `sbv2/tests/test_card.py` が
#: 両向きで見る。ここが要るのは「自分の契約以外を拒む」という 1 方向だけ。
FOREIGN_PIPELINE = "siglip2/1"


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


def _birefnet_manifest(model: str = MODEL) -> dict[str, Any]:
    """BiRefNet 系の最小 manifest（解像度以外は実物と重ならない偽値）。"""
    side = int(model)
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
                    "imageWidth": side,
                    "imageHeight": side,
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
        manifest["models"][MODEL]["pipeline"] = FOREIGN_PIPELINE
        with pytest.raises(ValueError, match=BIREFNET_SUPPORTED_PIPELINE):
            render_birefnet_model_card(manifest, REPO, CHECKPOINT)

    def test_it_refuses_a_checkpoint_it_cannot_attribute(self) -> None:
        """帰属表に無い checkpoint で描くと、`base_model` の無いカード（= 出所を名乗って
        いない再配布）が黙って出る。既知の一覧を添えて落とす。
        """
        with pytest.raises(ValueError, match="tiny"):
            render_birefnet_model_card(_birefnet_manifest(), REPO, "tiny")

    def test_it_refuses_a_resolution_it_has_not_measured(self) -> None:
        """カードが名乗る資源の数は 1 つ残らず実測に紐づく（実測の無い解像度では描かない）。"""
        manifest = _birefnet_manifest()
        manifest["models"][MODEL]["pipelineConfig"]["imageWidth"] = 512
        with pytest.raises(ValueError, match="512"):
            render_birefnet_model_card(manifest, REPO, CHECKPOINT)

    def test_it_renders_the_same_bytes_for_the_same_manifest(self) -> None:
        manifest = _birefnet_manifest()
        before = copy.deepcopy(manifest)
        assert render_birefnet_model_card(manifest, REPO, CHECKPOINT) == (
            render_birefnet_model_card(manifest, REPO, CHECKPOINT)
        )
        assert manifest == before

    def test_the_title_follows_the_checkpoint(self) -> None:
        """上流が名前で売っているモデルはその名前で呼ぶ（見出しは帰属表 1 つから来る）。"""
        assert "# Lucida (BiRefNet) — Karume" in render_birefnet_model_card(
            _birefnet_manifest(), REPO, "lucida"
        )
        assert "# BiRefNet HR — Karume" in render_birefnet_model_card(
            _birefnet_manifest(), REPO, CHECKPOINT
        )

    def test_the_attribution_does_not_follow_the_model_name(self) -> None:
        """モデル名は解像度で、帰属の軸ではない — 同じ manifest でも席が違えば別の上流を名乗る。"""
        for checkpoint, repo in BIREFNET_UPSTREAM.items():
            card = render_birefnet_model_card(_birefnet_manifest(), REPO, checkpoint)
            assert f"base_model: {repo}" in card

    def test_the_upstream_table_is_the_only_source_of_the_repository_ids(self) -> None:
        """`BIREFNET_UPSTREAM` は帰属表からの導出（2 表にすると片方だけ動ける）。"""
        assert {
            name: entry.repo for name, entry in BIREFNET_CHECKPOINTS.items()
        } == BIREFNET_UPSTREAM


class TestBirefnetEntryPoint:
    """カードが案内するロード入口 — `fromPretrained` の 1 本だけ。"""

    def test_it_does_not_advertise_the_local_asset_entry_point(self) -> None:
        """`fromAssets` は案内しない（2026-08-29 裁定）。

        分割配布形も読めるようになった（X2-101）が、あちらはバイト列を自分で持っている前提の
        ローカルデバッグ向けの面で、HF から使う読者の普通の入口は `fromPretrained`。両方を
        並べると「どちらを使うのか」を読者に判断させることになる。`fromPretrained` 側も併せて
        見るのは、Usage ごと消えても通る門にしないため。
        """
        card = render_birefnet_model_card(_birefnet_manifest(), REPO, CHECKPOINT)
        assert "fromAssets" not in card
        assert "BirefnetPipeline.fromPretrained" in card


class TestBirefnetResourceNote:
    """実行資源の注記（利用者が「渡す前に知りたい事実」— manifest に無いので定数）。"""

    @pytest.mark.parametrize("model", sorted(BIREFNET_RESOURCES))
    def test_the_card_names_the_gpu_memory_that_resolution_needs(self, model: str) -> None:
        """MUST: 総確保と、128MiB 既定を超える binding の両方を**そのモデルの実測で**名乗る。

        WebGPU の `maxStorageBufferBindingSize` の仕様既定は 128MiB なので、それを超える
        binding は「端末によっては要求自体が通らない」制約。カードが黙っていると、読み手は
        `requiredLimits` が空なこと（= 常駐分は既定内）を「既定スペックで動く」と読む。
        """
        card = render_birefnet_model_card(_birefnet_manifest(model), REPO, CHECKPOINT)
        resources = BIREFNET_RESOURCES[model]

        assert resources.total in card
        assert resources.binding in card
        assert resources.run in card
        assert BIREFNET_RESOURCE_MEASUREMENT in card
        assert "maxStorageBufferBindingSize" in card

    def test_it_does_not_carry_another_resolutions_measurement(self) -> None:
        """1 つの実測を全モデルへ写すと、片方のカードが**測っていない数**を名乗る。"""
        card = render_birefnet_model_card(_birefnet_manifest(MODEL), REPO, CHECKPOINT)

        assert BIREFNET_RESOURCES["2048"].total not in card
