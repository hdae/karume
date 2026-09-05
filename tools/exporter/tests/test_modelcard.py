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
from collections.abc import Sequence
from typing import Any

import pytest

from karume.modelcard import (
    CardMetadata,
    from_pretrained,
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
# `quants` / `models` / `require_pipeline` / `render` は wheel の外の
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
        "format": "karume/4",
        "generator": "karume/9.9.9",
        "defaultModel": "zeta",
        "models": {
            "zeta": {
                "pipeline": "fake/1",
                "weights": {
                    "front": {
                        "f16": {
                            "shards": [_ref("zeta/front-f16.safetensors", 4096, "a" * 64)],
                            "extras": {"rope": _ref("zeta/front-rope.safetensors", 64, "b" * 64)},
                        },
                        "i8": {"shards": [_ref("zeta/front-i8.safetensors", 2048, "c" * 64)]},
                    },
                    "tables": {
                        "f16": {"shards": [_ref("zeta/tables.safetensors", 128, "d" * 64)]},
                        "i8": {"shards": [_ref("zeta/tables.safetensors", 128, "d" * 64)]},
                    },
                },
                "assets": {"tokenizer": _ref("shared/tokenizer.json", 32, "e" * 64)},
                "quants": {
                    "f16": {"weights": {"front": "f16", "tables": "f16"}, "session": {}},
                    "w8": {
                        "weights": {"front": "i8", "tables": "i8"},
                        "session": {"linearCompute": "a8"},
                        "label": "Half size (int8)",
                        "description": "Both components in int8 storage.",
                    },
                },
                "defaultQuant": "w8",
                "pipelineConfig": {},
            },
            "alpha": {
                "pipeline": "fake/1",
                "weights": {
                    "front": {"f16": {"shards": [_ref("alpha/front.safetensors", 512, "f" * 64)]}}
                },
                "assets": {},
                "quants": {"f16": {"weights": {"front": "f16"}, "session": {}}},
                "defaultQuant": "f16",
                "pipelineConfig": {},
            },
        },
    }


def _row(lines: Sequence[str], name: str) -> str:
    """quant 表からその席の行を 1 本引く（列の位置ではなく席名で引く）。"""
    return next(line for line in lines if line.startswith(f"| `{name}`"))


class TestQuantsDownload:
    """Download 欄 = その席を選んだ読み手が実際に落とすバイト数（2026-09-03 裁定）。

    shard 上限 256 MiB でファイル本数が 3〜4 倍になり、shard 1 本 1 行のファイル表は廃止した。
    読み手が知りたい「このプリセットで何 GiB 落ちるか」を席ごとの合計 1 セルで持つ。
    """

    def test_it_sums_the_shards_extras_and_assets_the_quant_selects(self) -> None:
        """f16 = front 4,096 + rope 64 + tables 128 + tokenizer 32 = 4,320 B（= 4.22 KiB）。"""
        lines = quants(_manifest()["models"]["zeta"])

        assert "| 4.22 KiB (32 B shared) |" in _row(lines, "f16")

    def test_it_leaves_out_the_dtype_the_quant_does_not_select(self) -> None:
        """w8 = front i8 2,048 + tables 128 + tokenizer 32 = 2,208 B = 2.16 KiB（f16 の 4,096 は
        入らない）。

        席ごとに数え直していなければ、f16 側の重みを動かしたとき w8 の欄も動く。
        """
        manifest = _manifest()
        manifest["models"]["zeta"]["weights"]["front"]["f16"]["shards"][0]["size"] = 1 << 30

        before = _row(quants(_manifest()["models"]["zeta"]), "w8")
        after = _row(quants(manifest["models"]["zeta"]), "w8")
        assert "| 2.16 KiB (" in before
        assert after == before
        assert "| 1.00 GiB (" in _row(quants(manifest["models"]["zeta"]), "f16")

    def test_it_counts_the_extras_of_the_selected_dtype(self) -> None:
        """付帯資産（`extras`）も落ちるファイル — f16 だけが持つ rope 64 B が合計に入る。"""
        manifest = _manifest()
        del manifest["models"]["zeta"]["weights"]["front"]["f16"]["extras"]

        assert "| 4.16 KiB (" in _row(quants(manifest["models"]["zeta"]), "f16")

    def test_a_file_two_components_point_at_is_counted_once(self) -> None:
        """1 本のファイルを 2 席が指す形（1 本化済みの rope_base と同型）で二重に数えない。"""
        manifest = _manifest()
        front = manifest["models"]["zeta"]["weights"]["front"]["f16"]["shards"]
        manifest["models"]["zeta"]["weights"]["tables"]["f16"]["shards"] = front

        # front 4,096 + rope 64 + tokenizer 32 = 4,192 B（tables の 128 は同一 path なので消える）。
        assert "| 4.09 KiB (" in _row(quants(manifest["models"]["zeta"]), "f16")

    def test_it_calls_out_the_bytes_a_second_model_does_not_fetch_again(self) -> None:
        """`shared/` の分は「2 本目のモデルではこの分を落とさない」量として添える。"""
        assert "(32 B shared)" in _row(quants(_manifest()["models"]["zeta"]), "f16")

    def test_a_model_that_shares_nothing_keeps_the_cell_bare(self) -> None:
        """掛からない注記は出さない（0 B shared と綴ると、共有がある形に読める）。"""
        row = _row(quants(_manifest()["models"]["alpha"]), "f16")

        assert row.count("512 B") == 1
        assert "shared" not in row

    def test_bytes_borrowed_from_another_repository_count_as_shared(self) -> None:
        """越境参照（ADR 0038 §7）も 2 度は落ちない — 同じ「落とし直さない量」に入る。"""
        manifest = _manifest()
        manifest["models"]["alpha"]["weights"]["front"]["f16"]["shards"] = [
            {
                "repo": "hdae/karume-source",
                "revision": "9" * 40,
                "path": "source/front.safetensors",
                "size": 512,
                "sha256": "f" * 64,
            }
        ]

        assert "| 512 B (512 B shared) |" in _row(quants(manifest["models"]["alpha"]), "f16")

    def test_it_calls_out_assets_that_are_a_real_share_of_the_download(self) -> None:
        """assets は**ホスト側で読むだけ**（gemma4 の PLE sidecar）— Download と常駐量の差になる。

        w8 では tokenizer 32 B が 2,208 B の 1.4%（≥ 1%）なので内訳が出る。f16 では 4,320 B の
        0.7% なので出ない — どのカードにも書くと、数 MiB の tokenizer に読み手の目を向ける
        だけの行になる。
        """
        lines = quants(_manifest()["models"]["zeta"])

        assert "(32 B shared; 32 B of assets, read on the host)" in _row(lines, "w8")
        assert "of assets" not in _row(lines, "f16")


class TestDownloadSizeUnits:
    """単位は**丸めた後**の値で選ぶ（`3.24 GiB` / `248 MiB` — 単位を跨いでも精度が揃う）。

    丸める前に選ぶと、繰り上がる帯で `1024 MiB` という有効 3 桁でも単位でもない綴りが出る。
    繰り上がりの閾値は 1000 ではなく **1024**（= 1 段上の単位のちょうど 1）— `1000 MiB` は
    MiB として正しい綴りで、上げると「1 単位未満」の `0.977 GiB` になる。
    """

    @pytest.mark.parametrize(
        ("size", "spelled"),
        [
            # 繰り上がる帯（丸める前に単位を選ぶ実装ではここが `1024 MiB` / `1024 KiB` になる）。
            (1_073_741_823, "1.00 GiB"),
            (1_048_575, "1.00 MiB"),
            # 境界の逆側 — 丸めても 1024 に届かないので単位はそのまま。
            (1_073_217_535, "1023 MiB"),
            # 4 桁でも MiB として正しい帯（1 段上げると 1 単位未満の綴りになる）。
            (1_048_576_000, "1000 MiB"),
            # 既存の綴りが動かないことの対照（docstring の 2 例と、単位未満の生バイト）。
            (3_479_000_000, "3.24 GiB"),
            (260_000_000, "248 MiB"),
            (512, "512 B"),
        ],
    )
    def test_it_spells_the_download_cell_at_three_significant_digits(
        self, size: int, spelled: str
    ) -> None:
        manifest = _manifest()
        manifest["models"]["alpha"]["weights"]["front"]["f16"]["shards"][0]["size"] = size

        assert f"| {spelled} |" in _row(quants(manifest["models"]["alpha"]), "f16")


class TestQuantsNotes:
    """廃止したファイル表から引き継いだ注記 — **掛かるときだけ**出す（条件は manifest 由来）。"""

    def test_it_points_at_karume_json_for_the_per_file_values(self) -> None:
        """表から sha256 が消えた分、正本の所在は必ず残す（fetch 層の照合先）。"""
        lines = quants(_manifest()["models"]["zeta"])

        assert (
            "Per-file `size` and `sha256` live in `karume.json` — verify against that at the fetch"
            " layer." in lines
        )

    def test_a_shared_path_is_explained(self) -> None:
        """`(N shared)` の意味は本文の 1 文と対（注記が無いと何と共有なのか読めない）。"""
        assert any(
            line.startswith("A path under `shared/`")
            for line in quants(_manifest()["models"]["zeta"])
        )

    def test_a_model_without_a_shared_path_says_nothing_about_shared(self) -> None:
        assert not any(
            line.startswith("A path under `shared/`")
            for line in quants(_manifest()["models"]["alpha"])
        )

    def test_it_names_the_repository_borrowed_bytes_come_from(self) -> None:
        """指し先（リポ + pin した commit）まで出して初めて注記が事実になる。"""
        manifest = _manifest()
        manifest["models"]["alpha"]["weights"]["front"]["f16"]["shards"] = [
            {
                "repo": "hdae/karume-source",
                "revision": "9" * 40,
                "path": "source/front.safetensors",
                "size": 512,
                "sha256": "f" * 64,
            }
        ]
        lines = quants(manifest["models"]["alpha"])

        assert any(
            line.startswith(
                "Some components are fetched from"
                " [`hdae/karume-source`](https://huggingface.co/hdae/karume-source)"
                " at commit `9999999999999999…`"
            )
            for line in lines
        )

    def test_a_self_contained_model_says_nothing_about_other_repositories(self) -> None:
        assert not any(
            line.startswith("Some components are fetched from")
            for line in quants(_manifest()["models"]["zeta"])
        )

    def test_an_i4_component_flags_the_safetensors_dialect(self) -> None:
        """`I4` は公式仕様に無い語（docs/limitations.md）— 公式パーサで開けない事実を添える。"""
        manifest = _manifest()
        manifest["models"]["zeta"]["weights"]["front"]["i4"] = {
            "shards": [_ref("zeta/front-i4.safetensors", 1024, "1" * 64)]
        }

        assert any(
            line.startswith("A component stored as `i4`")
            for line in quants(manifest["models"]["zeta"])
        )

    def test_a_model_without_i4_says_nothing_about_the_dialect(self) -> None:
        """f16 / i8 だけの配布形は公式互換のまま — 掛からない注意書きを載せない。"""
        assert not any(
            line.startswith("A component stored as `i4`")
            for line in quants(_manifest()["models"]["zeta"])
        )


class TestQuantsSection:
    def test_the_default_mark_follows_default_quant(self) -> None:
        """既定マークは `defaultQuant` から導く（表の 1 行目でも先頭でもない）。"""
        lines = quants(_manifest()["models"]["zeta"])

        assert [line for line in lines if "(default)" in line] == [
            "| `w8` (default) | **Half size (int8)** — Both components in int8 storage."
            " | 2.16 KiB (32 B shared; 32 B of assets, read on the host) |"
            " `front` = `i8` / `tables` = `i8` | `linearCompute` = `a8` |"
        ]

    def test_a_seat_without_the_presentation_fields_keeps_an_empty_cell(self) -> None:
        """表示欄は optional（ADR 0075 決定 1）— 書いていない席は id をそのまま読ませる。"""
        lines = quants(_manifest()["models"]["alpha"])

        assert "| `f16` (default) | — | 512 B | `front` = `f16` | — |" in lines

    def test_it_writes_the_abbreviation_legend_only_when_one_is_given(self) -> None:
        """略称の対応は**カードに必ず出す**（ADR 0074 決定 4）が、略称のない family には出ない。

        トークンが weights 名そのものの family（irodori）で「`dit` は `dit`」の行が生えると、
        対応表そのものが読み飛ばされる注記になる。
        """
        model = _manifest()["models"]["zeta"]
        legend = "In a quant name, `tbl` is the `tables` component."

        assert legend in quants(model, abbreviations={"tbl": "tables"})
        assert not any(line.startswith("In a quant name") for line in quants(model))


class TestFromPretrained:
    """使い方スニペットの `fromPretrained` 呼び出し — **object ref 形** + revision の pin の席。

    source を文字列 1 本で綴っていた頃は revision を書く席そのものが無く、読み手は暗黙に
    `main` 追従になっていた（リポを上げ直した日に他人のアプリが壊れる）。
    """

    def test_it_passes_the_repository_as_an_object_and_keeps_the_options_second(self) -> None:
        lines = from_pretrained("AnimaPipeline", "hdae/karume-anima-turbo", ['  // quant: "f16",'])

        assert lines[0] == "using pipeline = await AnimaPipeline.fromPretrained({"
        assert lines[1] == '  repo: "hdae/karume-anima-turbo",'
        assert lines[-3:] == ["}, {", '  // quant: "f16",', "});"]

    def test_it_offers_the_pin_as_a_commented_out_line(self) -> None:
        """既定は `main` 追従のまま — 外すだけで pin できる形にする（2 本のサンプルに割らない）。"""
        lines = from_pretrained("AnimaPipeline", "hdae/x", [])

        assert '  // revision: "<full commit sha>",' in lines
        assert any("Pin a commit for reproducible builds" in line for line in lines)

    def test_the_disposable_spelling_is_the_callers_fact(self) -> None:
        """同期 dispose か非同期かは pipeline ごとの事実（ここでは選ばない）。"""
        lines = from_pretrained("Siglip2Pipeline", "hdae/x", [], disposable="await using")

        assert lines[0].startswith("await using pipeline = await Siglip2Pipeline")


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
        sections = [models(manifest), *model_sections(manifest, [quants])]
        return render(sections)

    def test_it_draws_no_file_table(self) -> None:
        """shard 1 本 1 行の表は廃止（2026-09-03 裁定）— 席ごとの合計が Download 欄に入る。"""
        card = self._card(_manifest())

        assert "### Files" not in card
        assert "| Key | Dtype | Path | Size | sha256 |" not in card
        assert "| Quant | What it is | Download | Weights | Compute |" in card

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
