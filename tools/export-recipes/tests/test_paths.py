"""recipe 全体のパス解決の根（`_shared.paths`）。

`REPO_ROOT = Path(__file__).resolve().parents[3]` は**ファイルの深さに依存する派生値**で、
1 段ずれると 7 定数が丸ごとずれる。破れ方が悪い: `INPUTS_ROOT` / `SERIES_ROOT` を引く家族
テストは資産不在の `skipif` / `importorskip` へ落ちるので、**全部 skip されたまま緑**になる。

MUST: `models/` / `outputs/` の**存在**は要求しない（どちらも git 追跡外で、資産の無い機に
存在しない）。ここが見るのは階層の数と綴りだけ。
"""

from __future__ import annotations

from _shared.paths import (
    BENCH_ROOT,
    DIST_ROOT,
    EXAMPLES_ROOT,
    INPUTS_ROOT,
    MISC_ROOT,
    OUTPUTS_ROOT,
    REPO_ROOT,
    SERIES_ROOT,
)


class TestRepoRoot:
    def test_it_points_at_the_repository_root(self) -> None:
        """リポ直下の恒久物（`deno.json` — CLAUDE.md 検証コマンド節）から根を検める。"""
        assert (REPO_ROOT / "deno.json").is_file()

    def test_it_can_reach_back_to_this_module_source(self) -> None:
        """自分自身へ戻れること = `parents[3]` の階層数そのものの直接の証明。"""
        assert (REPO_ROOT / "tools" / "export-recipes" / "_shared" / "paths.py").is_file()


class TestDerivedRoots:
    def test_the_three_asset_roots_hang_off_the_repository_root(self) -> None:
        """資産 3 根（配布形 / 生成物 / 入力素材）はリポ直下（docs/assets-layout.md）。"""
        assert DIST_ROOT.parent == REPO_ROOT
        assert OUTPUTS_ROOT.parent == REPO_ROOT
        assert INPUTS_ROOT.parent == REPO_ROOT

    def test_the_generated_roots_hang_off_the_outputs_root(self) -> None:
        """MUST: 生成物は `models/` に置かない（paths.py 冒頭の DECIDED）。"""
        for root in (SERIES_ROOT, BENCH_ROOT, MISC_ROOT, EXAMPLES_ROOT):
            assert root.parent == OUTPUTS_ROOT, root

    def test_the_spellings_are_the_ones_the_layout_document_names(self) -> None:
        """綴りは書き手（台本）と読み手（dist ドライバ）の共有知識なので値まで固定する。"""
        assert DIST_ROOT.name == "models"
        assert OUTPUTS_ROOT.name == "outputs"
        assert INPUTS_ROOT.name == "inputs"
        assert SERIES_ROOT.name == "series"
