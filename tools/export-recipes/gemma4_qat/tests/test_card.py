"""QAT 配布形のモデルカード描画（`gemma4_qat.distribution.render_card`）。

実物の配布形の `karume.json` は使わない — 偽 manifest の値がそのまま本文に出ること
（＝手書きの数値が 1 つも混ざっていないこと）を見るのがここの仕事なので、実物と**違う**値で
組んだほうが検出力が高い。

ここが持つのはカード側にしか無い門 — テンプレートの pipeline 固有性、帰属が上流の
mobile QAT チェックポイントを名指ししていること、そして「固定値の保持を再量子化と呼ばない」
という本文の主張である。
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from gemma4_qat.config import checkpoint_name
from gemma4_qat.distribution import render_card

#: 使い方スニペットに綴られるリポ ID（pipeline の宣言から dist が渡す）。
REPO = "hdae/fake-repo"

#: 別 family の pipeline 契約（**綴りをそのまま持つ** — recipe 間のコード結合を作らないため）。
FOREIGN_PIPELINE = "gemma4/1"

QAT_PIPELINE = "gemma4-qat/1"

#: quant の表示欄（**実物と重ならない綴り** — 手書きが混ざっていれば表に出ない）。
FAKE_QUANT_LABEL = "Fake fixed seat"
FAKE_QUANT_DESCRIPTION = "Only this sentence may appear in the quant table."


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


def _qat_manifest(model: str = "e2b") -> dict[str, Any]:
    """QAT の最小 manifest（値は実物と重ならない偽値）。"""
    return {
        "format": "karume/4",
        "generator": "karume/9.9.9",
        "defaultModel": model,
        "models": {
            model: {
                "pipeline": QAT_PIPELINE,
                "weights": {
                    "model": {"i4": _container(_ref("model/model.i4.safetensors", 13, "a"))}
                },
                "assets": {"ple_index": _ref("ple/ple.json", 7, "b")},
                "quants": {
                    "i4": {
                        "weights": {"model": "i4"},
                        "session": {},
                        "label": FAKE_QUANT_LABEL,
                        "description": FAKE_QUANT_DESCRIPTION,
                    }
                },
                "defaultQuant": "i4",
                "pipelineConfig": {"chunkLength": 3, "capacity": 5, "maxPosition": 7},
            }
        },
    }


class TestQatCardGate:
    def test_it_refuses_a_pipeline_it_does_not_describe(self) -> None:
        manifest = _qat_manifest()
        manifest["models"]["e2b"]["pipeline"] = FOREIGN_PIPELINE
        with pytest.raises(ValueError, match=QAT_PIPELINE):
            render_card(manifest, REPO)

    def test_it_refuses_a_model_it_cannot_attribute(self) -> None:
        """帰属表に無いモデル名で描くと、上流を名乗っていない再配布が黙って出る。"""
        with pytest.raises(ValueError, match="未対応"):
            render_card(_qat_manifest("12b"), REPO)

    def test_it_renders_the_same_bytes_for_the_same_manifest(self) -> None:
        manifest = _qat_manifest()
        before = copy.deepcopy(manifest)

        assert render_card(manifest, REPO) == render_card(manifest, REPO)
        assert manifest == before


class TestQatCardBody:
    def test_it_names_the_upstream_mobile_checkpoint(self) -> None:
        card = render_card(_qat_manifest(), REPO)

        assert f"google/{checkpoint_name('e2b')}" in card
        assert "gemma-4-E2B-it-qat-mobile-transformers" in card

    def test_it_does_not_call_preserved_integers_a_requantization(self) -> None:
        """固定値の保持を再量子化と呼ばない（`render_card` の docstring の主張）。"""
        card = render_card(_qat_manifest(), REPO)

        assert "re-quantized" not in card
        assert "Fixed integers and scales are preserved" in card

    def test_it_takes_the_quant_presentation_from_the_manifest(self) -> None:
        """席の説明は manifest の値がそのまま出る（カード側に写しを持たない）。"""
        card = render_card(_qat_manifest(), REPO)

        assert FAKE_QUANT_LABEL in card
        assert FAKE_QUANT_DESCRIPTION in card
