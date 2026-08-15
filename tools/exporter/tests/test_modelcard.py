"""モデルカードの**汎用描画部品**（`karume.modelcard`）。

実物の配布形の `karume.json` は使わない — 偽 manifest の値がそのまま本文に出ること
（＝手書きの数値が 1 つも混ざっていないこと）を見るのがここの仕事なので、実物と**違う**値で
組んだほうが検出力が高い。

NOTE: pipeline 別のカードテンプレートは 1 つ残らず wheel の外へ出た（ADR 0065 段 3+4 完了）
ので、テンプレート単位のケースは `tools/export-recipes/<family>/tests/test_card.py` に居る。
ここに残るのは、テンプレートを経由せずに core だけで観測できる層。
"""

from __future__ import annotations

from typing import Any

from karume.modelcard import CardMetadata, frontmatter

# ---- frontmatter の任意席 -----------------------------------------------------
#
# recipe 側の family はどれも `base_model_relation` / `license_name` / `license_link` を
# 持たない（格納形を変えない配布形 + SPDX 標準タグ）か、持っていても wheel の外なので、
# **書いたときに出る**側の枝は family 経由では 1 つも踏まれない。席そのものは core の責務
# なので、ここで直接見る。


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
        lines = frontmatter(self._metadata())
        for absent in ("base_model_relation", "license_name", "license_link"):
            assert not any(line.startswith(absent) for line in lines), absent

    def test_it_writes_each_optional_field_that_is_set(self) -> None:
        lines = frontmatter(
            self._metadata(
                base_model_relation="quantized",
                license_name="owner-terms",
                license_link="https://example.invalid/terms",
            )
        )
        assert "base_model_relation: quantized" in lines
        assert "license_name: owner-terms" in lines
        assert "license_link: https://example.invalid/terms" in lines
