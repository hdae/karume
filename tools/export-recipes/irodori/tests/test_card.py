"""Irodori 配布形のモデルカード描画（`irodori.card`）— カード側にしか無い門。

実物の配布形の `karume.json` は使わない — 偽 manifest の値がそのまま本文に出ること
（＝手書きの数値が 1 つも混ざっていないこと）を見るのがここの仕事なので、実物と**違う**値で
組んだほうが検出力が高い。

manifest からの導出（表・数・使い方）は `irodori/tests/test_distribution.py` の
`TestIrodoriModelCard` が**組み立て 1 周ぶん**で見る。これまで Irodori のテンプレートは
core の `tests/test_modelcard.py` に固有の節を持っていなかったので `test_card.py` を作って
いなかったが、「案内するロード入口」は組み立てを 1 周しなくても観測できるカード側だけの門なので
ここに置く（2026-08-30）。
"""

from __future__ import annotations

from typing import Any

from irodori.card import IRODORI_SUPPORTED_PIPELINE, render_irodori_model_card

#: 使い方スニペットに綴られるリポ ID（組み立て先のディレクトリ名から dist が渡す）。
REPO = "hdae/fake-repo"


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


def _irodori_manifest() -> dict[str, Any]:
    """Irodori の最小 manifest（値は実物と重ならない偽値）。"""
    return {
        "format": "karume/3",
        "generator": "karume/9.9.9",
        "defaultModel": "ZA",
        "models": {
            "ZA": {
                "pipeline": IRODORI_SUPPORTED_PIPELINE,
                "weights": {
                    "backbone": {"f16": {"shards": [_ref("ZA/backbone/model.f16.st", 11, "a")]}},
                    "dit": {"f16": {"shards": [_ref("ZA/dit/model.f16.st", 13, "b")]}},
                },
                "assets": {},
                "quants": {
                    "f16": {"weights": {"backbone": "f16", "dit": "f16"}, "session": {}},
                },
                "defaultQuant": "f16",
                "pipelineConfig": {
                    "maxTextLen": 17,
                    "maxCaptionLen": 19,
                    "speakerRows": 23,
                    "ditSymMax": 29,
                    "frameRate": 7,
                    "sampleRate": 31,
                    "hopLength": 37,
                    "codecHaloFrames": 3,
                    "latentDim": 41,
                    "speakerPatchSize": 5,
                    "speakerDim": 43,
                    "textDim": 47,
                    "captionDim": 53,
                    "timestepEmbedDim": 59,
                    "steps": 61,
                    "initScale": 0.111,
                    "cfgMinT": 0.25,
                    "cfgMaxT": 0.75,
                    "cfgScales": {"text": 1.5, "speaker": 2.5, "caption": 3.5},
                    "minSeconds": 0.5,
                    "maxSeconds": 4.0,
                    "speakerUncondMode": "mask",
                    "cfgGuidanceMode": "independent",
                },
            }
        },
    }


class TestIrodoriEntryPoint:
    """カードが案内するロード入口 — `fromPretrained` の 1 本だけ。"""

    def test_it_does_not_advertise_the_local_asset_entry_point(self) -> None:
        """`fromAssets` は案内しない（2026-08-29 裁定）。

        分割配布形も読めるようになった（X2-101）が、あちらはバイト列を自分で持っている前提の
        ローカルデバッグ向けの面で、HF から使う読者の普通の入口は `fromPretrained`。両方を
        並べると「どちらを使うのか」を読者に判断させることになる。`fromPretrained` 側も併せて
        見るのは、Usage ごと消えても通る門にしないため。
        """
        card = render_irodori_model_card(_irodori_manifest(), REPO)
        assert "fromAssets" not in card
        assert "IrodoriPipeline.fromPretrained" in card
