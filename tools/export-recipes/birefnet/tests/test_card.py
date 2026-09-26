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
    BIREFNET_MATTE_WEIGHTS,
    BIREFNET_RESOURCE_MEASUREMENT,
    BIREFNET_RESOURCE_QUANT,
    BIREFNET_RESOURCES,
    BIREFNET_SUPPORTED_PIPELINE,
    BIREFNET_UPSTREAM,
    render_birefnet_model_card,
)

#: 使い方スニペットに綴られるリポ ID（pipeline の宣言から dist が渡す）。
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


def _container(*refs: dict[str, Any]) -> dict[str, Any]:
    """weights の 1 dtype ぶん（`karume/5` の `container` — ADR 0109 決定 3）。

    `descriptor` はカードが読まない欄なので、形だけ実物どおりに置く（2 文書の長さと sha256）。
    """
    return {
        "container": {
            "descriptor": {
                "graph": {"length": 40, "sha256": "0" * 64},
                "model": {"length": 24, "sha256": "1" * 64},
            },
            "parts": list(refs),
        }
    }


def _birefnet_manifest(model: str = MODEL) -> dict[str, Any]:
    """BiRefNet 系の最小 manifest（解像度以外は実物と重ならない偽値）。"""
    side = int(model)
    return {
        "format": "karume/5",
        "generator": "karume/9.9.9",
        "defaultModel": model,
        "models": {
            model: {
                "pipeline": BIREFNET_SUPPORTED_PIPELINE,
                "weights": {
                    "matte": {"f32": _container(_ref("m/model.f32-00001-of-00003.krm", 13, "d"))}
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

    def test_the_total_is_named_as_the_measured_seats_number(self) -> None:
        """席が 2 つ並ぶカードで、総確保がどの席の実測かを言う（f16 席の総確保は未測 —
        2026-09-26 レビュー F2）。"""
        card = render_birefnet_model_card(_with_seat(_birefnet_manifest(), "f16"), REPO, CHECKPOINT)
        prose = " ".join(card.split())

        assert BIREFNET_RESOURCE_QUANT == "f32"
        assert f"{BIREFNET_RESOURCES[MODEL].total} allocated in total with the `f32` quant" in prose

    def test_it_refuses_a_model_without_the_measured_seat(self) -> None:
        """実測した席が無い配布形では、総確保を名乗れない（別の席の数として読まれる）。"""
        manifest = _birefnet_manifest()
        model = manifest["models"][MODEL]
        model["quants"] = {"f16": {"weights": {"matte": "f16"}, "session": {}}}
        model["weights"]["matte"] = {"f16": model["weights"]["matte"]["f32"]}
        model["defaultQuant"] = "f16"

        with pytest.raises(ValueError, match="実測した quant 'f32'"):
            render_birefnet_model_card(manifest, REPO, CHECKPOINT)

    def test_it_does_not_carry_another_resolutions_measurement(self) -> None:
        """1 つの実測を全モデルへ写すと、片方のカードが**測っていない数**を名乗る。"""
        card = render_birefnet_model_card(_birefnet_manifest(MODEL), REPO, CHECKPOINT)

        assert BIREFNET_RESOURCES["2048"].total not in card


def _with_seat(manifest: dict[str, Any], dtype: str, *, default: bool = False) -> dict[str, Any]:
    """既定モデルへ `dtype` の weights と同名の quant 席を足す（`default` なら既定席にする）。"""
    model = manifest["models"][manifest["defaultModel"]]
    model["weights"]["matte"][dtype] = _container(
        _ref(f"m/model.{dtype}-00001-of-00002.krm", 7, "e")
    )
    model["quants"][dtype] = {"weights": {"matte": dtype}, "session": {}}
    if default:
        model["defaultQuant"] = dtype
    return manifest


class TestBirefnetStorageAttribution:
    """席ごとの格納の説明と `base_model_relation`（ADR 0113 — checkpoint の格納から決まる）。"""

    def test_the_weights_key_is_the_distributions_role(self) -> None:
        """カードは循環 import を避けて weights キーを自分で綴る — 配布 recipe と一致させる。"""
        from birefnet.distribution import BIREFNET_ROLE

        assert BIREFNET_MATTE_WEIGHTS == BIREFNET_ROLE

    @pytest.mark.parametrize("checkpoint", sorted(BIREFNET_CHECKPOINTS))
    def test_an_f32_default_declares_no_relation(self, checkpoint: str) -> None:
        """f32 席は checkpoint の値を変えない — 4 値のどれでもないので置かない。"""
        card = render_birefnet_model_card(_with_seat(_birefnet_manifest(), "f16"), REPO, checkpoint)
        assert "base_model_relation" not in card

    def test_an_f16_default_for_the_f16_checkpoint_is_not_quantized(self) -> None:
        """HR の checkpoint 自身が f16 — f16 席を既定にしても値は変わらない。"""
        card = render_birefnet_model_card(
            _with_seat(_birefnet_manifest(), "f16", default=True), REPO, "hr"
        )
        assert "base_model_relation" not in card

    def test_an_f16_default_for_the_f32_checkpoint_is_quantized(self) -> None:
        card = render_birefnet_model_card(
            _with_seat(_birefnet_manifest(), "f16", default=True), REPO, "lucida"
        )
        assert "base_model_relation: quantized" in card

    def test_the_storage_lines_follow_the_seats_in_the_manifest(self) -> None:
        """dtype ラベルは manifest から — f32 だけの配布形は f16 席を名乗らない。"""
        plain = render_birefnet_model_card(_birefnet_manifest(), REPO, "hr")
        both = render_birefnet_model_card(_with_seat(_birefnet_manifest(), "f16"), REPO, "hr")

        assert "  - `f32`:" in plain and "  - `f16`:" not in plain
        assert "  - `f32`:" in both and "  - `f16`:" in both

    def test_the_hr_card_names_its_f16_checkpoint(self) -> None:
        """旧文面「source checkpoint's own f32 values」は HR では事実と違った（上流は f16）。"""
        card = render_birefnet_model_card(_with_seat(_birefnet_manifest(), "f16"), REPO, "hr")

        assert "**No quantization**" in card
        assert "widened exactly from f16 to f32" in card
        assert "own f32 values" not in card

    def test_the_lucida_card_declares_its_f16_seat_quantized(self) -> None:
        card = render_birefnet_model_card(_with_seat(_birefnet_manifest(), "f16"), REPO, "lucida")

        assert "**Quantization**" in card
        assert "  - `f16`: **quantized**" in card
        assert "  - `f32`: the checkpoint's own f32 values, unchanged." in card

    def test_a_seat_it_has_no_wording_for_is_refused(self) -> None:
        """段 2 の i8 は文面と一緒に足す — 推し量った文面を帰属節へ載せない。"""
        with pytest.raises(ValueError, match="i8"):
            render_birefnet_model_card(_with_seat(_birefnet_manifest(), "i8"), REPO, "hr")
