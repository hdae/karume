"""リポの dist ドライバ（`tools/export-recipes/dist.py`）— 受理集合と置き場の既定の合成。

core の {@link karume.dist.PIPELINES} は wheel だけで組める分しか持たない（ADR 0065 決定 2）。
ここで固定するのは「recipe 側の family が 1 つ残らず表に載り、旧 UX の既定が保たれる」こと —
載せ忘れは `--pipeline <family>` が「そんな pipeline は無い」で落ちる形でしか表面化せず、
資産を作り終えた後に初めて気づく。

置き場の既定（`models/` / `outputs/series/`）もドライバの持ち物（ADR 0065 Consequences）—
core は綴りを持たないので、{@link dist.default_out_dir} の規則はここが見る。
"""

from __future__ import annotations

import http.client
import io
import re
import subprocess
import sys
import urllib.error
from pathlib import Path
from typing import Any, ClassVar

import pytest

import dist
from _shared.paths import DIST_ROOT, REPO_ROOT, SERIES_ROOT
from anima.distribution import EXTRA_PIPELINE as ANIMA_EXTRA_PIPELINE
from anima.distribution import OFFICIAL_PIPELINE as ANIMA_OFFICIAL_PIPELINE
from birefnet.distribution import LUCIDA_PIPELINE
from birefnet.distribution import PIPELINE as BIREFNET_PIPELINE
from depth_anything.distribution import PIPELINE as DEPTH_ANYTHING_PIPELINE
from gemma4.distribution import PIPELINE as GEMMA4_PIPELINE
from gemma4.distribution import QAT_Q4_0_PIPELINE as GEMMA4_QAT_Q4_0_PIPELINE
from gemma4_qat.distribution import PIPELINE as GEMMA4_QAT_PIPELINE
from irodori.distribution import PIPELINE as IRODORI_PIPELINE
from karume.dist import PIPELINES as CORE_PIPELINES
from karume.dist import (
    CrossRepoReference,
    DistError,
    ExternalComponents,
    declared_references,
    resolve_repo,
)
from moss.distribution import PIPELINE as MOSS_PIPELINE
from sbv2.distribution import FN_PIPELINE as SBV2_FN_PIPELINE
from sbv2.distribution import PIPELINE as SBV2_PIPELINE
from siglip2.distribution import PIPELINE as SIGLIP2_PIPELINE
from vowel_detector.distribution import PIPELINE as VOWEL_DETECTOR_PIPELINE
from wan.distribution import PIPELINE as WAN_PIPELINE
from wan.distribution import TI2V_PIPELINE as WAN_TI2V_PIPELINE
from wan.umt5_distribution import PIPELINE as UMT5_PIPELINE

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
    "gemma4-qat": GEMMA4_QAT_Q4_0_PIPELINE,
    "gemma4-qat-mobile": GEMMA4_QAT_PIPELINE,
    "wan": WAN_PIPELINE,
    "wan-ti2v": WAN_TI2V_PIPELINE,
    "umt5": UMT5_PIPELINE,
    "moss-image": MOSS_PIPELINE,
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
            "gemma4-qat-mobile",
            "siglip2",
            "birefnet",
            "lucida",
            "depth-anything",
            "wan",
            "wan-ti2v",
            "umt5",
            "moss-image",
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
        """複数モデルの出力先は `--out` の明示を求める（1 モデルの宣言だけを引く — 並べた
        モデルの宣言が揃うかは突き合わせない）。"""
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
    # The q4_0 QAT line (`--pipeline gemma4-qat` → karume-gemma4-qat) and the mobile QAT line
    # (`--pipeline gemma4-qat-mobile` → karume-gemma4-qat-mobile, renamed 2026-11-04) have no
    # published revision yet, so neither has a TS table entry.
    ("irodori", ["v4-small"], None, "karume-irodori-v4-small"),
    ("irodori", ["v4.1-small"], None, "karume-irodori-v4.1-small"),
    # `--repo` 無しで公開リポを指す（Pipeline がファミリーのリポ名を宣言する — W-RC5-2）。
    ("sbv2", ["F1", "F2", "M1", "M2"], None, "karume-sbv2-jvnv"),
    ("siglip2", ["base", "so400m"], None, "karume-siglip2"),
    ("umt5", ["xxl"], None, "karume-umt5-xxl"),
    # Wan2.1（`--pipeline wan` → karume-wan2.1）は HF に公開しないので載せない
    # （ADR 0118 追記 2026-10-15）。
    ("wan-ti2v", ["ti2v-5b"], None, "karume-wan2.2"),
    ("moss-image", ["sft"], None, "karume-moss-image"),
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
        """公開 revision の正本（TS の各 family の `SOURCES`）と同じリポの集合を焼くこと。

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


class TestDeclaredReferencesOfThePublishedRepositories:
    """The cross-repository references the recipes declare, read against the published set
    ({@link RELEASE_REPOSITORIES}) — ADR 0038 §7 addendum 2026-10-28."""

    @pytest.mark.parametrize(("pipeline", "models", "repo", "directory"), RELEASE_REPOSITORIES)
    def test_every_reference_points_at_another_published_repository(
        self, pipeline: str, models: list[str], repo: str | None, directory: str
    ) -> None:
        """A referenced repository must be uploaded first (release-runbook §0), so it is one of
        the published ones — and never the repository being built."""
        published = {name for *_, name in RELEASE_REPOSITORIES}
        for reference in declared_references(dist.PIPELINES[pipeline], models):
            assert reference.repo_name in published - {directory}, reference

    def test_the_three_referencing_repositories(self) -> None:
        referencing = {
            directory: sorted(
                reference.repo_name
                for reference in declared_references(dist.PIPELINES[pipeline], models)
            )
            for pipeline, models, _, directory in RELEASE_REPOSITORIES
        }
        assert {name: repos for name, repos in referencing.items() if repos} == {
            "karume-anima-extra": ["karume-anima"],
            "karume-irodori-v4.1-small": ["karume-irodori-v4-small"],
            "karume-wan2.2": ["karume-umt5-xxl"],
        }


#: A reference of the synthetic cases below.
_REFERENCE = CrossRepoReference("karume-source", "m", ("w",))
_SHA = "0123456789abcdef0123456789abcdef01234567"


def _no_hub(url: str) -> tuple[Any, str | None]:
    raise AssertionError(f"no Hub access expected: {url}")


class TestReferencePins:
    """The revision source of a build ({@link dist.reference_pins}) — one per build."""

    def test_explicit_revisions_are_left_to_the_core(self) -> None:
        assert dist.reference_pins(
            (_REFERENCE,),
            {"karume-source": _SHA},
            resolve=None,
            allow_placeholder=False,
            get=_no_hub,
        ) == ({}, None)

    def test_the_placeholder_flag_pins_every_reference_at_the_placeholder(self) -> None:
        assert dist.reference_pins(
            (_REFERENCE,), {}, resolve=None, allow_placeholder=True, get=_no_hub
        ) == ({"karume-source": dist.PLACEHOLDER_REVISION}, None)

    def test_an_explicit_placeholder_is_refused(self) -> None:
        with pytest.raises(DistError, match=dist.ALLOW_PLACEHOLDER_FLAG):
            dist.reference_pins(
                (_REFERENCE,),
                {"karume-source": dist.PLACEHOLDER_REVISION},
                resolve=None,
                allow_placeholder=False,
                get=_no_hub,
            )

    def test_an_explicit_placeholder_with_the_flag_is_told_the_sources_exclude_each_other(
        self,
    ) -> None:
        """Not "use the flag instead" — the caller already passed it."""
        with pytest.raises(DistError, match="exclude each other"):
            dist.reference_pins(
                (_REFERENCE,),
                {"karume-source": dist.PLACEHOLDER_REVISION},
                resolve=None,
                allow_placeholder=True,
                get=_no_hub,
            )

    def test_a_declared_reference_without_a_source_is_refused(self) -> None:
        with pytest.raises(DistError, match=r"reference \['karume-source'\] across repositories"):
            dist.reference_pins(
                (_REFERENCE,), {}, resolve=None, allow_placeholder=False, get=_no_hub
            )

    @pytest.mark.parametrize(
        ("explicit", "resolve", "allow_placeholder"),
        [
            ({"karume-source": _SHA}, "hf", False),
            ({"karume-source": _SHA}, None, True),
            ({}, "hf", True),
        ],
    )
    def test_two_sources_exclude_each_other(
        self, explicit: dict[str, str], resolve: str | None, allow_placeholder: bool
    ) -> None:
        with pytest.raises(DistError, match="exclude each other"):
            dist.reference_pins(
                (_REFERENCE,),
                explicit,
                resolve=resolve,
                allow_placeholder=allow_placeholder,
                get=_no_hub,
            )

    @pytest.mark.parametrize(
        ("resolve", "allow_placeholder", "flag"),
        [("hf", False, "--resolve-refs"), (None, True, "--allow-placeholder-ref")],
    )
    def test_a_flag_for_a_build_that_references_nothing_is_refused(
        self, resolve: str | None, allow_placeholder: bool, flag: str
    ) -> None:
        with pytest.raises(DistError, match=f"{flag} was given, but the models built declare no"):
            dist.reference_pins(
                (), {}, resolve=resolve, allow_placeholder=allow_placeholder, get=_no_hub
            )

    def test_the_hub_pins_the_main_of_every_referenced_repository(self, tmp_path: Path) -> None:
        requested: list[str] = []

        def get(url: str) -> tuple[Any, str | None]:
            requested.append(url)
            return ({"sha": _SHA}, None) if url.endswith("/revision/main") else ([], None)

        revisions, check = dist.reference_pins(
            (_REFERENCE,), {}, resolve="hf", allow_placeholder=False, get=get
        )

        assert revisions == {"karume-source": _SHA}
        assert check is not None
        assert requested == [
            "https://huggingface.co/api/models/hdae/karume-source/revision/main",
            f"https://huggingface.co/api/models/hdae/karume-source/tree/{_SHA}?recursive=true",
        ]


class TestReadingTheHub:
    """The HF API side of `--resolve-refs hf` — mocked; the tests read no network."""

    def test_it_reads_every_page_of_the_tree_at_the_resolved_commit(self) -> None:
        api = "https://huggingface.co/api/models/hdae/karume-source"
        pages = {
            f"{api}/revision/main": ({"sha": _SHA}, None),
            f"{api}/tree/{_SHA}?recursive=true": (
                [{"type": "directory", "path": "m"}, {"type": "file", "path": "m/a", "oid": "1"}],
                f"{api}/tree/{_SHA}?recursive=true&cursor=2",
            ),
            f"{api}/tree/{_SHA}?recursive=true&cursor=2": (
                [{"type": "file", "path": "m/b", "oid": "2"}],
                None,
            ),
        }

        commit = dist.read_hub_main("hdae/karume-source", pages.__getitem__)

        assert commit.revision == _SHA
        assert sorted(commit.files) == ["m/a", "m/b"]

    @pytest.mark.parametrize("body", [{}, {"sha": "main"}, [], {"sha": "A" * 40}])
    def test_a_main_without_a_commit_sha_fails(self, body: Any) -> None:
        with pytest.raises(DistError, match="gave no commit sha"):
            dist.read_hub_main("hdae/karume-source", lambda _url: (body, None))

    def test_the_real_reader_follows_the_link_header_and_reports_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Response(io.BytesIO):
            headers: ClassVar[dict[str, str]] = {
                "Link": '<https://huggingface.co/api/next?cursor=x>; rel="next"'
            }

        monkeypatch.setattr(
            dist.urllib.request, "urlopen", lambda _request, timeout: Response(b"[1]")
        )
        assert dist.hf_get("https://huggingface.co/api/x") == (
            [1],
            "https://huggingface.co/api/next?cursor=x",
        )

        def offline(_request: object, timeout: float) -> io.BytesIO:
            raise urllib.error.URLError("unreachable")

        monkeypatch.setattr(dist.urllib.request, "urlopen", offline)
        with pytest.raises(DistError, match=r"could not read .* offline, pin the revision"):
            dist.hf_get("https://huggingface.co/api/x")

    def test_a_connection_dropped_while_reading_the_body_fails_the_same_way(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`http.client.IncompleteRead` is not an `OSError` — it still gets the offline advice."""

        class Dropped(io.BytesIO):
            headers: ClassVar[dict[str, str]] = {}

            def read(self, size: int | None = -1) -> bytes:
                raise http.client.IncompleteRead(b"[1", 10)

        monkeypatch.setattr(dist.urllib.request, "urlopen", lambda _request, timeout: Dropped())
        with pytest.raises(DistError, match=r"could not read .* offline, pin the revision"):
            dist.hf_get("https://huggingface.co/api/x")

    def test_the_git_blob_id_matches_git(self, tmp_path: Path) -> None:
        """The values `git hash-object` gives for an empty file and for `hello\\n`."""
        (tmp_path / "empty").write_bytes(b"")
        (tmp_path / "hello").write_bytes(b"hello\n")

        assert dist.git_blob_id(tmp_path / "empty") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
        assert dist.git_blob_id(tmp_path / "hello") == "ce013625030ba8dba906f756967f9e9ca394464a"


class TestConfirmingOnTheHub:
    """{@link dist.confirm_on_hub} — every borrowed file is in the pinned commit with the local
    bytes."""

    def _case(self, tmp_path: Path) -> tuple[ExternalComponents, list[dict[str, Any]]]:
        (tmp_path / "m").mkdir()
        (tmp_path / "m" / "large.krm").write_bytes(b"large")
        (tmp_path / "m" / "small.krm").write_bytes(b"hello\n")
        components = ExternalComponents(
            repo="hdae/karume-source", revision=_SHA, dist=tmp_path, model="m", roles=("w",)
        )
        refs = [
            {
                "repo": components.repo,
                "revision": _SHA,
                "path": "m/large.krm",
                "size": 5,
                "sha256": "a" * 64,
            },
            {
                "repo": components.repo,
                "revision": _SHA,
                "path": "m/small.krm",
                "size": 6,
                "sha256": "b" * 64,
            },
        ]
        return components, refs

    def _commit(self, files: dict[str, dict[str, Any]]) -> dict[str, dist.HubCommit]:
        return {"hdae/karume-source": dist.HubCommit("hdae/karume-source", _SHA, files)}

    _LARGE: ClassVar[dict[str, Any]] = {"oid": "1" * 40, "lfs": {"oid": "a" * 64, "size": 5}}
    _SMALL: ClassVar[dict[str, Any]] = {"oid": "ce013625030ba8dba906f756967f9e9ca394464a"}

    def test_an_lfs_oid_and_a_git_blob_that_match_pass(self, tmp_path: Path) -> None:
        components, refs = self._case(tmp_path)
        commit = self._commit({"m/large.krm": self._LARGE, "m/small.krm": self._SMALL})

        dist.confirm_on_hub(commit, components, refs)

    @pytest.mark.parametrize(
        ("files", "message"),
        [
            (
                {"m/large.krm": {"oid": "1" * 40, "lfs": {"oid": "c" * 64}}},
                rf"m/large.krm: LFS oid {'c' * 64} ≠ local sha256 {'a' * 64}",
            ),
            ({"m/large.krm": {"oid": "1" * 40, "lfs": {"oid": "a" * 64}}}, "m/small.krm: not in"),
            (
                {
                    "m/large.krm": {"oid": "1" * 40, "lfs": {"oid": "a" * 64}},
                    "m/small.krm": {"oid": "2" * 40},
                },
                "m/small.krm: git blob 2{40} ≠ local ce0136",
            ),
        ],
    )
    def test_a_missing_or_different_file_fails_naming_it(
        self, tmp_path: Path, files: dict[str, dict[str, Any]], message: str
    ) -> None:
        components, refs = self._case(tmp_path)

        with pytest.raises(DistError, match=message):
            dist.confirm_on_hub(self._commit(files), components, refs)


class TestCoreArguments:
    """The driver-only options never reach the core's parser."""

    def test_the_driver_options_are_removed_with_their_values(self) -> None:
        assert dist.core_arguments(
            [
                "--pipeline",
                "irodori",
                "--resolve-refs",
                "hf",
                "--resolve-refs=hf",
                "--allow-placeholder-ref",
                "--intake",
                "x",
                "--ref-revision",
                "a=b",
            ]
        ) == ["--pipeline", "irodori", "--ref-revision", "a=b"]


#: README の受理集合を綴る 1 文（`--pipeline` の引数として叩ける名前がバッククォートで並ぶ）。
#: 同じ節の「13 pipeline seats across 10 families」は**家族数**を語る別の文なので拾わない。
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
