"""モデルカードの**汎用描画部品**（`karume.modelcard`）。

実物の配布形の `karume.json` は使わない — 偽 manifest の値がそのまま本文に出ること
（＝手書きの数値が 1 つも混ざっていないこと）を見るのがここの仕事なので、実物と**違う**値で
組んだほうが検出力が高い。

NOTE: pipeline 別のカードテンプレートは 1 つ残らず wheel の外へ出た（ADR 0065 段 3+4 完了）
ので、テンプレート単位のケースは `tools/export-recipes/<family>/tests/test_card.py` に居る。
ここに残るのは、テンプレートを経由せずに core だけで観測できる層。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from karume.modelcard import (
    CardMetadata,
    file_rows,
    files,
    frontmatter,
    model_sections,
    models,
    quants,
    render,
    require_pipeline,
)

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


# ---- recipe 向けの公開描画部品 -------------------------------------------------
#
# `file_rows` / `files` / `quants` / `models` / `require_pipeline` / `render` は wheel の外の
# `card.py` が名指しで呼ぶ公開面（ADR 0065 段 6）。core だけを回したときにこの面の回帰が
# 見えるよう、偽 manifest を被験体として core 側に門を置く（recipe 側の門は family ごとの
# 間接的な踏み方で、回帰の主語が読めない）。


def _ref(path: str, size: int, sha: str) -> dict[str, Any]:
    """ファイル参照の 3 点セット（実物と**違う**値 — 手書き混入はここで見える）。"""
    return {"path": path, "size": size, "sha256": sha}


def _manifest() -> dict[str, Any]:
    """2 モデル・共有資産つきの偽 manifest（ADR 0041 §2 の形）。

    モデルの並びは `zeta` → `alpha` で辞書順と**逆** — 並べ替えが混ざれば表の順で分かる。
    `tables` は f16 / i8 の両 dtype が同一 path を指す形（rope_base の 1 本化と同型）。
    """
    return {
        "format": "karume/2",
        "generator": "karume/9.9.9",
        "defaultModel": "zeta",
        "models": {
            "zeta": {
                "pipeline": "fake/1",
                "weights": {
                    "front": {
                        "f16": {
                            "file": _ref("zeta/front-f16.safetensors", 4096, "a" * 64),
                            "extras": {"rope": _ref("zeta/front-rope.safetensors", 64, "b" * 64)},
                        },
                        "i8": {"file": _ref("zeta/front-i8.safetensors", 2048, "c" * 64)},
                    },
                    "tables": {
                        "f16": {"file": _ref("zeta/tables.safetensors", 128, "d" * 64)},
                        "i8": {"file": _ref("zeta/tables.safetensors", 128, "d" * 64)},
                    },
                },
                "assets": {"tokenizer": _ref("shared/tokenizer.json", 32, "e" * 64)},
                "quants": {
                    "f16": {"weights": {"front": "f16", "tables": "f16"}, "session": {}},
                    "w8": {
                        "weights": {"front": "i8", "tables": "i8"},
                        "session": {"linearCompute": "i8a8"},
                    },
                },
                "defaultQuant": "w8",
                "pipelineConfig": {},
            },
            "alpha": {
                "pipeline": "fake/1",
                "weights": {
                    "front": {"f16": {"file": _ref("alpha/front.safetensors", 512, "f" * 64)}}
                },
                "assets": {},
                "quants": {"f16": {"weights": {"front": "f16"}, "session": {}}},
                "defaultQuant": "f16",
                "pipelineConfig": {},
            },
        },
    }


class TestFileRows:
    """行は **path で一意化**する（現物にない 2 行目を表に生やさない）。"""

    def test_two_dtypes_that_point_at_one_path_fold_into_a_single_row(self) -> None:
        rows = file_rows(_manifest()["models"]["zeta"])

        folded = [
            (key, labels) for key, labels, ref in rows if ref["path"].endswith("tables.safetensors")
        ]
        assert folded == [("tables", ["f16", "i8"])]
        # 席は path で 1 つだけ（front f16 / front i8 / rope / tables / tokenizer）。
        assert [ref["path"] for _, _, ref in rows] == [
            "zeta/front-f16.safetensors",
            "zeta/front-i8.safetensors",
            "zeta/front-rope.safetensors",
            "zeta/tables.safetensors",
            "shared/tokenizer.json",
        ]


class TestFilesSection:
    def test_a_shared_path_is_listed_and_explained(self) -> None:
        """`shared/` の行は本文の注記と対（注記だけ残って行が消えると読み手が迷子になる）。"""
        lines = files(_manifest()["models"]["zeta"])

        assert any("`shared/tokenizer.json`" in line for line in lines)
        assert any(line.startswith("A path under `shared/`") for line in lines)


class TestQuantsSection:
    def test_the_default_mark_follows_default_quant(self) -> None:
        """既定マークは `defaultQuant` から導く（表の 1 行目でも先頭でもない）。"""
        lines = quants(_manifest()["models"]["zeta"])

        assert [line for line in lines if "(default)" in line] == [
            "| `w8` (default) | `front` = `i8` / `tables` = `i8` | `linearCompute` = `i8a8` |"
        ]


class TestRequirePipeline:
    def test_it_refuses_a_manifest_that_mixes_another_pipeline(self) -> None:
        """1 つでも別契約なら描かない — 表は合っているのに説明だけ別のモデル、を作らない。"""
        manifest = _manifest()
        manifest["models"]["alpha"]["pipeline"] = "other/1"

        with pytest.raises(ValueError, match="'alpha' の pipeline 'other/1'"):
            require_pipeline(manifest, "fake/1")

    def test_it_accepts_a_manifest_where_every_model_matches(self) -> None:
        require_pipeline(_manifest(), "fake/1")


class TestRenderDeterminism:
    """同一 manifest なら**バイト単位で同一**（差分 = 資産が変わった、が再組み立ての前提）。"""

    @staticmethod
    def _card(manifest: dict[str, Any]) -> str:
        sections = [models(manifest), *model_sections(manifest, [files, quants])]
        return render(sections)

    def test_two_renderings_of_an_equal_manifest_agree_byte_for_byte(self) -> None:
        # JSON 経由で組み直した別オブジェクト（同値・同じ挿入順）— 同一 dict の再描画だけだと
        # 「オブジェクトの同一性に依存した並び」を踏めない。
        rebuilt = json.loads(json.dumps(_manifest()))

        assert self._card(_manifest()) == self._card(rebuilt)

    def test_the_body_ends_with_exactly_one_newline(self) -> None:
        card = self._card(_manifest())

        assert card.endswith("\n") and not card.endswith("\n\n")

    def test_the_models_keep_the_manifest_order(self) -> None:
        """並べ替えを挟むと「manifest の並び」という事実がカードから消える。"""
        card = self._card(_manifest())

        assert card.index("## Model: zeta") < card.index("## Model: alpha")
