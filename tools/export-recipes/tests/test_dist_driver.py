"""リポの dist ドライバ（`tools/export-recipes/dist.py`）— 受理集合と置き場の既定の合成。

core の {@link karume.dist.PIPELINES} は wheel だけで組める分しか持たない（ADR 0065 決定 2）。
ここで固定するのは「recipe 側の family が 1 つ残らず表に載り、旧 UX の既定が保たれる」こと —
載せ忘れは `--pipeline <family>` が「そんな pipeline は無い」で落ちる形でしか表面化せず、
資産を作り終えた後に初めて気づく。

置き場の既定（`models/` / `outputs/series/`）もドライバの持ち物（ADR 0065 Consequences）—
core は綴りを持たないので、{@link dist.default_out_dir} の規則はここが見る。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

import dist
from _shared.paths import DIST_ROOT, REPO_ROOT, SERIES_ROOT
from anima.distribution import EXTRA_PIPELINE as ANIMA_EXTRA_PIPELINE
from anima.distribution import OFFICIAL_PIPELINE as ANIMA_OFFICIAL_PIPELINE
from birefnet.distribution import LUCIDA_PIPELINE
from birefnet.distribution import PIPELINE as BIREFNET_PIPELINE
from depth_anything.distribution import PIPELINE as DEPTH_ANYTHING_PIPELINE
from gemma4.distribution import PIPELINE as GEMMA4_PIPELINE
from gemma4_qat.distribution import PIPELINE as GEMMA4_QAT_PIPELINE
from irodori.distribution import PIPELINE as IRODORI_PIPELINE
from karume.dist import PIPELINES as CORE_PIPELINES
from karume.dist import DistError, resolve_repo
from sbv2.distribution import FN_PIPELINE as SBV2_FN_PIPELINE
from sbv2.distribution import PIPELINE as SBV2_PIPELINE
from siglip2.distribution import PIPELINE as SIGLIP2_PIPELINE
from vowel_detector.distribution import PIPELINE as VOWEL_DETECTOR_PIPELINE

#: 配布 recipe を持つ family の全量（名前 → その family が公開する `PIPELINE`）。
#: **ここが受理集合の期待値**で、`dist.PIPELINES` の載せ忘れも余剰も 1 つの表で検出する。
RECIPE_PIPELINES = {
    "anima": ANIMA_OFFICIAL_PIPELINE,
    "anima-extra": ANIMA_EXTRA_PIPELINE,
    "sbv2": SBV2_PIPELINE,
    "sbv2-fn": SBV2_FN_PIPELINE,
    "irodori": IRODORI_PIPELINE,
    "siglip2": SIGLIP2_PIPELINE,
    "birefnet": BIREFNET_PIPELINE,
    "lucida": LUCIDA_PIPELINE,
    "depth-anything": DEPTH_ANYTHING_PIPELINE,
    "vowel-detector": VOWEL_DETECTOR_PIPELINE,
    "gemma4": GEMMA4_PIPELINE,
    "gemma4-qat": GEMMA4_QAT_PIPELINE,
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

    def test_the_two_birefnet_repositories_stay_separate_pipelines(self) -> None:
        """BiRefNet HR と派生の Lucida も**別の席**（ADR 0092 決定 1）。

        MIT の著作権行はリポごとに違う（Lucida は fine-tune 側と上流の 2 行）ので、1 つに
        畳むとどちらかのリポが自分のものでない著作権を名乗るか、上流の表示を落とす。
        """
        base = dist.PIPELINES["birefnet"]
        derived = dist.PIPELINES["lucida"]

        assert base is not derived
        assert base.root_files["LICENSE.md"] != derived.root_files["LICENSE.md"]
        assert base.root_files["NOTICE.md"] != derived.root_files["NOTICE.md"]

    def test_the_two_sbv2_voice_families_stay_separate_pipelines(self) -> None:
        """SBV2 の JVNV 系（`sbv2`）と FN 系（`sbv2-fn`）も**別の席**。

        ライセンスがファミリーごとに違う（JVNV = CC BY-SA 4.0・FN = Booth の頒布条件で同梱
        できる条文が無い）ので、1 つに畳むと FN のリポが CC BY-SA の条文と JVNV の帰属を名乗る。
        """
        jvnv = dist.PIPELINES["sbv2"]
        fn = dist.PIPELINES["sbv2-fn"]

        assert jvnv is not fn
        assert sorted(jvnv.card_profiles) == ["jvnv"]
        assert sorted(fn.card_profiles) == ["fn"]
        assert "Attribution-ShareAlike 4.0" in jvnv.root_files["LICENSE.md"]
        assert fn.root_files == {}

    def test_every_distribution_pipeline_ships_its_legal_text(self) -> None:
        """配布リポ直下に `LICENSE.md` と `NOTICE.md` の**両方**を置く席（ADR 0092 決定 7）。

        名指しにするのは、新しい family が黙って同梱なしで生えるのを「表に載せ忘れた」形で
        見えるようにするため。

        vowel-detector はここに入らない: MIT が要求するのは「全文 + 著作権行」だけで、
        改変告知（`NOTICE.md`）の宛先が無い（上流と著作権者が同じ）。許諾表示そのものは
        下の `test_the_mit_only_pipeline_still_ships_its_permission_notice` が見る。
        `sbv2-fn` も入らない: HF 公開が保留で、上流の書面条件（Booth の頒布ページ）に同梱
        できるライセンス文が無い（{@link sbv2.distribution.FN_PIPELINE}）。
        """
        expected = {
            "anima",
            "anima-extra",
            "sbv2",
            "irodori",
            "gemma4",
            "gemma4-qat",
            "siglip2",
            "birefnet",
            "lucida",
            "depth-anything",
        }
        carried = {
            name
            for name, spec in dist.PIPELINES.items()
            if set(spec.root_files) == {"LICENSE.md", "NOTICE.md"}
        }
        assert carried == expected

    def test_the_mit_only_pipeline_still_ships_its_permission_notice(self) -> None:
        """MUST: MIT の席も許諾表示は落とさない（`NOTICE.md` が無いことと別の話）。

        `karume.dist.LEGAL_PATHS` は受理集合であって**在ることを要求しない**ので、
        `root_files` が空のまま組み上がっても `verify_dist` も manifest 検査も何も言わない。
        """
        root_files = dist.PIPELINES["vowel-detector"].root_files

        assert set(root_files) == {"LICENSE.md"}
        assert "MIT License" in root_files["LICENSE.md"]
        assert "Copyright (c) " in root_files["LICENSE.md"]

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
        assert dist.default_out_dir(SIGLIP2_PIPELINE, ["base"]).name == "karume-siglip2"

    def test_it_refuses_to_invent_a_family_repository_name(self) -> None:
        """ファミリーリポの名前（例 `karume-sbv2-jvnv`）はモデル名の並びからは決まらない。"""
        with pytest.raises(DistError, match="--out"):
            dist.default_out_dir(SBV2_PIPELINE, ["jvnv-F1", "jvnv-F2"])


#: 公開中のリポを焼く手順（pipeline・`--model` の並び・`--repo`）→ 焼き先 `models/<名前>/`。
#: `tools/release/hf-upload.zsh` はこのディレクトリを `hdae/<名前>` へ上げるので、カードの
#: Usage 例がこの名前を指さないと公開カードが別のリポを指す。
RELEASE_REPOSITORIES = [
    (
        "anima",
        [
            "anima-turbo-v1.1",
            "anima-v1.0",
            "anima-aesthetic-v1.1",
            "anima-turbo-v1.0",
            "anima-aesthetic-v1.0",
        ],
        None,
        "karume-anima",
    ),
    ("anima-extra", ["anima-wai-v1.0", "anima-copycat-20260610"], None, "karume-anima-extra"),
    ("birefnet", ["1024", "2048"], None, "karume-birefnet-hr"),
    ("lucida", ["1024", "2048"], None, "karume-lucida"),
    ("depth-anything", ["small"], None, "karume-depth-anything-v2"),
    ("gemma4", ["e2b"], None, "karume-gemma4"),
    ("irodori", ["v4-small"], None, "karume-irodori-v4-small"),
    ("irodori", ["v4.1-small"], None, "karume-irodori-v4.1-small"),
    ("sbv2", ["F1", "F2", "M1", "M2"], "hdae/karume-sbv2-jvnv", "karume-sbv2-jvnv"),
    ("siglip2", ["base", "so400m"], None, "karume-siglip2"),
]


class TestReleaseRepositories:
    """公開リポの焼き直しで、カードの Usage 例が上げ先のリポ名のまま変わらないこと。

    repo の出所が出力ディレクトリ名から pipeline の宣言へ移っても、手順どおりの `--out`
    （`models/<リポ名>/`）で焼いたカードは同じバイトになる（違うのは別名で焼いたときだけ）。
    """

    @pytest.mark.parametrize(("pipeline", "models", "repo", "directory"), RELEASE_REPOSITORIES)
    def test_the_card_names_the_repository_it_is_uploaded_to(
        self, pipeline: str, models: list[str], repo: str | None, directory: str
    ) -> None:
        assert resolve_repo(dist.PIPELINES[pipeline], models, repo) == f"hdae/{directory}"

    def test_the_table_names_the_same_repositories_as_the_typescript_sources(self) -> None:
        """公開 revision の正本（TS の各 family の `*_SOURCES`）と同じリポの集合を焼くこと。

        2 つの表は別の言語で独立に更新されるので、リポの追加・改名を片側だけにすると、カードの
        Usage 例と pin のリポが割れる。TS 側は Python から読めるテキストなので機械で突き合わせる。
        """
        sources = sorted((REPO_ROOT / "packages" / "models" / "src").glob("*/config.ts"))
        assert sources, "packages/models/src/*/config.ts が見つからない（置き場が動いた）"
        declared = {
            match
            for path in sources
            for match in re.findall(
                r'^\s*repo: "(hdae/[^"]+)",$', path.read_text(encoding="utf-8"), re.MULTILINE
            )
        }

        assert {f"hdae/{directory}" for *_, directory in RELEASE_REPOSITORIES} == declared


#: README の受理集合を綴る 1 文（`--pipeline` の引数として叩ける名前がバッククォートで並ぶ）。
#: 同じ節の「10 pipeline seats across 8 families」は**家族数**を語る別の文なので拾わない。
README_ACCEPTED_SET = re.compile(r"The accepted set is (?P<names>[^.]+)\.")


class TestReadme:
    """README（英語 MUST）の受理集合が実装から乖離しないこと。

    README どおりに `--pipeline <名前>` を叩いて argparse が落ちる状態は、利用者が最初に
    踏む地面が抜けている形。ADR 0092 決定 5 でこの README はライセンス carve-out の宣言
    文書でもあるので、他の記述の信頼性がそのまま carve-out の信頼性に見える。
    """

    @staticmethod
    def _accepted_names() -> list[str]:
        readme = (Path(dist.__file__).resolve().parent / "README.md").read_text(encoding="utf-8")
        match = README_ACCEPTED_SET.search(readme)
        assert match is not None, "README の受理集合の 1 文が見つからない（綴りが動いた）"
        return re.findall(r"`([^`]+)`", match.group("names"))

    def test_the_readme_lists_every_accepted_pipeline(self) -> None:
        assert sorted(self._accepted_names()) == sorted(dist.PIPELINES)

    def test_the_readme_lists_them_in_the_help_order(self) -> None:
        """並びは `--help` の並び（既定が先頭）— 読み手が CLI と突き合わせられる形にする。"""
        assert self._accepted_names() == list(dist.PIPELINES)


def _torch_loaded_after(source: str) -> bool:
    """新しいインタプリタで `source` を実行し、torch が `sys.modules` に入ったかを返す。

    「どのモジュールが既に import 済みか」はテストセッションの `sys.modules` では判定できない
    （他のテストが torch を読み込み済み）ので、core の test_package_init と同じ subprocess 形で
    毎回確かめる。cwd を recipe のルートにして、`dist` / family を pytest と同じ綴りで引く。
    """
    completed = subprocess.run(
        [sys.executable, "-c", f"import sys\n{source}\nprint('torch' in sys.modules)"],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(dist.__file__).resolve().parent,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip() == "True"


class TestImportingTheDriver:
    """配布・カード層は torch を読まずに組めること（siglip2 / depth の `measurements.py` の前提）。

    patch 層（torch 依存）を配布経路の import 連鎖から外す判断は、「`import dist` は torch を
    読まない」を根拠にしている。どれか 1 つの distribution / card が patch 層や torch 依存の
    helper を module 直下で import すると、この性質は黙って崩れる。
    """

    def test_importing_the_driver_does_not_load_torch(self) -> None:
        assert not _torch_loaded_after("import dist")

    def test_importing_a_patch_layer_does_load_torch(self) -> None:
        """対照: 同じ判定が torch 依存の import を検出できる（上の検査が恒真でない）。"""
        assert _torch_loaded_after("import siglip2.patch")
