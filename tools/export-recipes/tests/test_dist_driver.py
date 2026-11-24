"""The repository's dist driver (`tools/export-recipes/dist.py`) — how the accepted set and the
default locations are composed.

The core's {@link karume.dist.PIPELINES} holds only what the wheel alone can build (ADR 0065
decision 2). What is pinned here is "every recipe-side family is in the table, and the defaults of
the old UX are kept" — a missing entry only surfaces as `--pipeline <family>` failing with "no such
pipeline", noticed for the first time after the assets are built.

The default locations (`models/` / `outputs/series/`) also belong to the driver (ADR 0065
Consequences) — the core holds no spelling for them, so the rule of {@link dist.default_out_dir} is
checked here.
"""

from __future__ import annotations

import http.client
import io
import json
import re
import subprocess
import sys
import urllib.error
from pathlib import Path
from typing import Any, ClassVar

import pytest

import dist
from _shared.paths import DIST_ROOT, REPO_ROOT, SERIES_ROOT
from anima.distribution import ANIMA_2_9B_PIPELINE
from anima.distribution import EXTRA_PIPELINE as ANIMA_EXTRA_PIPELINE
from anima.distribution import OFFICIAL_PIPELINE as ANIMA_OFFICIAL_PIPELINE
from birefnet.distribution import LUCIDA_PIPELINE
from birefnet.distribution import PIPELINE as BIREFNET_PIPELINE
from depth_anything.distribution import PIPELINE as DEPTH_ANYTHING_PIPELINE
from embeddinggemma2.distribution import PIPELINE as EMBEDDINGGEMMA2_PIPELINE
from gemma4.distribution import PIPELINE as GEMMA4_PIPELINE
from gemma4.distribution import QAT_Q4_0_PIPELINE as GEMMA4_QAT_Q4_0_PIPELINE
from gemma4_qat.distribution import PIPELINE as GEMMA4_QAT_PIPELINE
from irodori.distribution import LARGE_PIPELINE as IRODORI_LARGE_PIPELINE
from irodori.distribution import PIPELINE as IRODORI_PIPELINE
from karume.dist import (
    MANIFEST_FILENAME,
    CrossRepoReference,
    DistError,
    ExternalComponents,
    assert_name_map,
    declared_references,
    resolve_repo,
)
from karume.dist import PIPELINES as CORE_PIPELINES
from karume.migrate import card_format_statement
from karume.modelcard import HF_OWNER, UNOFFICIAL_CONVERSION_NOTICE
from moss.distribution import PIPELINE as MOSS_PIPELINE
from sbv2.distribution import PIPELINE as SBV2_PIPELINE
from siglip2.distribution import PIPELINE as SIGLIP2_PIPELINE
from wan.distribution import TI2V_PIPELINE as WAN_TI2V_PIPELINE
from wan.umt5_distribution import PIPELINE as UMT5_PIPELINE

#: Every family that has a distribution recipe (name → the `PIPELINE` that family exports).
#: **This is the expected accepted set**; one table catches both a missing and a surplus entry of
#: `dist.PIPELINES`.
RECIPE_PIPELINES = {
    "anima": ANIMA_OFFICIAL_PIPELINE,
    "anima-extra": ANIMA_EXTRA_PIPELINE,
    "anima-2.9b": ANIMA_2_9B_PIPELINE,
    "sbv2": SBV2_PIPELINE,
    "irodori": IRODORI_PIPELINE,
    "irodori-large": IRODORI_LARGE_PIPELINE,
    "siglip2": SIGLIP2_PIPELINE,
    "birefnet": BIREFNET_PIPELINE,
    "lucida": LUCIDA_PIPELINE,
    "depth-anything": DEPTH_ANYTHING_PIPELINE,
    "gemma4": GEMMA4_PIPELINE,
    "gemma4-qat": GEMMA4_QAT_Q4_0_PIPELINE,
    "gemma4-qat-mobile": GEMMA4_QAT_PIPELINE,
    "wan-ti2v": WAN_TI2V_PIPELINE,
    "umt5": UMT5_PIPELINE,
    "moss-image": MOSS_PIPELINE,
    "embeddinggemma-2": EMBEDDINGGEMMA2_PIPELINE,
}


class TestRegistry:
    def test_it_carries_every_core_pipeline(self) -> None:
        for name, spec in CORE_PIPELINES.items():
            assert dist.PIPELINES[name] is spec

    def test_it_adds_every_recipe_pipeline(self) -> None:
        for name, spec in RECIPE_PIPELINES.items():
            assert dist.PIPELINES[name] is spec

    def test_the_table_holds_nothing_else(self) -> None:
        """A surplus entry is "a pipeline in no family's table" — never something that builds from
        a name alone."""
        assert sorted(dist.PIPELINES) == sorted({**RECIPE_PIPELINES, **CORE_PIPELINES})

    def test_the_core_table_is_empty_now_that_every_family_has_moved(self) -> None:
        """If a family that has moved were still on the core side, the composition would have 2
        tables claim the same name.

        This is the very completion condition of ADR 0065 steps 3+4 — no family knowledge is left
        in the core wheel.
        """
        assert CORE_PIPELINES == {}

    def test_the_default_is_anima(self) -> None:
        """Keep the UX of the old `karume dist` (no arguments) on the driver side."""
        assert dist.DEFAULT_PIPELINE == "anima"
        assert dist.DEFAULT_PIPELINE in dist.PIPELINES

    def test_the_three_anima_repositories_stay_separate_pipelines(self) -> None:
        """The official models (`anima`), the fine-tunes (`anima-extra`) and the layer-expanded
        Anima-2.9B (`anima-2.9b`) are **separate entries** (the split axis of ADR 0087 and its
        addendum of 2026-11-24).

        `root_files` is one fixed set per Pipeline, so folding two into one would make one
        repository's modification notice disagree with its contents — it still reads as valid
        prose, so `verify_dist` and the manifest check both pass, and nobody notices until it is
        distributed.
        """
        pipelines = [dist.PIPELINES[name] for name in ("anima", "anima-extra", "anima-2.9b")]

        assert len({id(pipeline) for pipeline in pipelines}) == 3
        assert len({pipeline.root_files["NOTICE.md"] for pipeline in pipelines}) == 3

    def test_every_anima_repository_and_no_other_declare_the_recipe_name_map(self) -> None:
        """Anima's name map (`tools/export-recipes/anima/name-map.json`) goes into each of its
        repositories (ADR 0153); no other family has one yet, so none declares one."""
        recipe_map = (REPO_ROOT / "tools" / "export-recipes" / "anima" / "name-map.json").resolve()

        declaring = {
            name: pipeline.name_map.resolve()
            for name, pipeline in dist.PIPELINES.items()
            if pipeline.name_map is not None
        }

        assert declaring == {
            "anima": recipe_map,
            "anima-extra": recipe_map,
            "anima-2.9b": recipe_map,
        }
        assert_name_map(recipe_map)

    def test_the_two_birefnet_repositories_stay_separate_pipelines(self) -> None:
        """BiRefNet HR and its derivative Lucida are **separate entries** too (ADR 0092 decision
        1).

        The MIT copyright lines differ per repository (Lucida has 2 lines: the fine-tune's and the
        upstream's), so folding them into one would make one repository claim a copyright that is
        not its own, or drop the upstream notice.
        """
        base = dist.PIPELINES["birefnet"]
        derived = dist.PIPELINES["lucida"]

        assert base is not derived
        assert base.root_files["LICENSE.md"] != derived.root_files["LICENSE.md"]
        assert base.root_files["NOTICE.md"] != derived.root_files["NOTICE.md"]

    def test_every_distribution_pipeline_ships_its_legal_text(self) -> None:
        """The entries that put **both** `LICENSE.md` and `NOTICE.md` at the root of the
        distribution repository (ADR 0092 decision 7).

        They are named one by one so that a new family appearing silently without the bundled texts
        shows up as "missing from the table".
        """
        expected = {
            "anima",
            "anima-extra",
            "anima-2.9b",
            "sbv2",
            "irodori",
            "irodori-large",
            "gemma4",
            "gemma4-qat",
            "gemma4-qat-mobile",
            "siglip2",
            "birefnet",
            "lucida",
            "depth-anything",
            "wan-ti2v",
            "umt5",
            "moss-image",
            "embeddinggemma-2",
        }
        carried = {
            name
            for name, spec in dist.PIPELINES.items()
            if set(spec.root_files) == {"LICENSE.md", "NOTICE.md"}
        }
        assert carried == expected

    def test_every_pipeline_renders_its_own_model_card(self) -> None:
        """A card is a per-pipeline template — its renderer refuses another pipeline's
        manifest."""
        for name, spec in dist.PIPELINES.items():
            manifest = {"models": {"m": {"pipeline": f"{name}/0"}}}
            for render_card in spec.card_profiles.values():
                with pytest.raises(ValueError):
                    render_card(manifest, "hdae/x")


class TestDefaultPlaces:
    """The defaults of `--series` / `--out` — only the driver knows the repo topology."""

    def test_the_series_default_is_the_repository_series_root(self) -> None:
        """A run without `--series` reads the repository's `outputs/series/`."""
        assert dist.SERIES_ROOT is SERIES_ROOT

    def test_the_default_output_directory_follows_the_single_model(self) -> None:
        assert dist.default_out_dir(SBV2_PIPELINE, ["jvnv-F1"]).parent == DIST_ROOT
        assert dist.default_out_dir(SIGLIP2_PIPELINE, ["base"]).name == "karume-siglip2"

    def test_it_refuses_to_invent_a_family_repository_name(self) -> None:
        """The output directory of several models requires an explicit `--out` (only one model's
        declaration is looked up — whether the declarations of the listed models agree is not
        matched)."""
        with pytest.raises(DistError, match="--out"):
            dist.default_out_dir(SBV2_PIPELINE, ["jvnv-F1", "jvnv-F2"])


#: How each published repository is built (pipeline, the `--model` list, `--repo`) → the output
#: directory `models/<name>/`. `tools/release/hf-upload.zsh` uploads this directory to
#: `hdae/<name>`, so unless the card's Usage example names it, the published card points at
#: another repository. A repository not published yet (`karume-embeddinggemma-2`,
#: `karume-anima-2.9b`) is not listed: the TS `SOURCES` it is matched against get its entry at the
#: first publication (docs/release-runbook.md §3).
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
    ("gemma4-qat", ["e2b", "e4b", "12b"], None, "karume-gemma4-qat"),
    # The mobile QAT line, renamed from karume-gemma4-qat on 2026-11-04.
    ("gemma4-qat-mobile", ["e2b", "e4b"], None, "karume-gemma4-qat-mobile"),
    ("irodori", ["v4-small"], None, "karume-irodori-v4-small"),
    ("irodori", ["v4.1-small"], None, "karume-irodori-v4.1-small"),
    ("irodori", ["v4.1-small-mf"], None, "karume-irodori-v4.1-small-mf"),
    # The v4-Large line has its own Pipeline (the Gemma legal texts at the repository root).
    ("irodori-large", ["v4-large"], None, "karume-irodori-v4-large"),
    # Points at the published repository without `--repo` (the Pipeline declares the family's
    # repository name — W-RC5-2).
    ("sbv2", ["F1", "F2", "M1", "M2"], None, "karume-sbv2-jvnv"),
    ("siglip2", ["base", "so400m"], None, "karume-siglip2"),
    ("umt5", ["xxl"], None, "karume-umt5-xxl"),
    ("wan-ti2v", ["ti2v-5b"], None, "karume-wan2.2"),
    ("moss-image", ["sft"], None, "karume-moss-image"),
]


class TestReleaseRepositories:
    """Rebuilding a published repository keeps the card's Usage example on the name of the
    repository it is uploaded to.

    Even though the repo now comes from the pipeline's declaration instead of the output directory
    name, a card built with the documented `--out` (`models/<repository name>/`) has the same bytes
    (it differs only when built under another name).
    """

    @pytest.mark.parametrize(("pipeline", "models", "repo", "directory"), RELEASE_REPOSITORIES)
    def test_the_card_names_the_repository_it_is_uploaded_to(
        self, pipeline: str, models: list[str], repo: str | None, directory: str
    ) -> None:
        assert resolve_repo(dist.PIPELINES[pipeline], models, repo) == f"hdae/{directory}"

    def test_the_table_names_the_same_repositories_as_the_typescript_sources(self) -> None:
        """Build the same set of repositories as the source of truth for published revisions
        (each TS family's `SOURCES`).

        The 2 tables are updated independently in different languages, so adding or renaming a
        repository on one side only would split the card's Usage example from the pinned
        repository. The TS side is text Python can read, so they are matched mechanically.
        """
        sources = sorted((REPO_ROOT / "packages" / "models" / "src").glob("*/config.ts"))
        assert sources, "packages/models/src/*/config.ts not found (the location moved)"
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

    def test_the_five_referencing_repositories(self) -> None:
        referencing = {
            directory: sorted(
                reference.repo_name
                for reference in declared_references(dist.PIPELINES[pipeline], models)
            )
            for pipeline, models, _, directory in RELEASE_REPOSITORIES
        }
        assert {name: repos for name, repos in referencing.items() if repos} == {
            "karume-anima-extra": ["karume-anima"],
            "karume-irodori-v4-large": ["karume-irodori-v4-small"],
            "karume-irodori-v4.1-small": ["karume-irodori-v4-small"],
            "karume-irodori-v4.1-small-mf": [
                "karume-irodori-v4-small",
                "karume-irodori-v4.1-small",
            ],
            "karume-wan2.2": ["karume-umt5-xxl"],
        }


#: The house non-endorsement sentence as a pattern (its two slots filled by anything).
HOUSE_SENTENCE = re.compile(
    re.escape(UNOFFICIAL_CONVERSION_NOTICE)
    .replace(re.escape("{upstream}"), "(?P<upstream>.+)")
    .replace(re.escape("{author}"), "(?P<author>.+)")
)


def _mirrors_built_by(name: str) -> list[Path]:
    """The local distribution mirrors `--pipeline name` builds: those whose models the pipeline
    names after the mirror's own repository (`resolve_repo` — how the card names its repository)."""
    mirrors = []
    for manifest_path in sorted(DIST_ROOT.glob(f"*/{MANIFEST_FILENAME}")):
        models = list(json.loads(manifest_path.read_text(encoding="utf-8"))["models"])
        try:
            repo = resolve_repo(dist.PIPELINES[name], models, None)
        except (DistError, KeyError, ValueError):
            continue  # models this pipeline does not build
        if repo == f"{HF_OWNER}/{manifest_path.parent.name}":
            mirrors.append(manifest_path)
    return mirrors


class TestTheHouseSentenceOnEveryCard:
    """Every card opens with the house non-endorsement sentence, right under its title
    (maintainer's decision of 2026-11-04), drawn from each local mirror's own manifest — every
    pipeline of the registry, every attribution profile."""

    @pytest.mark.parametrize("name", sorted(dist.PIPELINES))
    def test_the_card_carries_the_sentence_once_under_its_title(self, name: str) -> None:
        mirrors = _mirrors_built_by(name)
        if not mirrors:
            pytest.skip(f"no distribution mirror under {DIST_ROOT} is built by --pipeline {name}")
        for manifest_path in mirrors:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for render_card in dist.PIPELINES[name].card_profiles.values():
                lines = render_card(manifest, f"{HF_OWNER}/{manifest_path.parent.name}").split("\n")
                sentences = [line for line in lines if HOUSE_SENTENCE.fullmatch(line)]
                title = next(index for index, line in enumerate(lines) if line.startswith("# "))
                assert sentences == [lines[title + 2]], manifest_path


#: The pipelines whose card does not state the manifest format (the upgrade leaves it as it is).
CARDS_WITHOUT_THE_FORMAT = frozenset({"gemma4-qat-mobile"})


class TestTheFormatStatementOnEveryCard:
    """Every card that names the manifest format does it once, in the one statement the upgrade of
    a repository rewrites (`karume.migrate.card_format_statement` — ADR 0153), drawn from each
    local mirror's own manifest."""

    @pytest.mark.parametrize("name", sorted(dist.PIPELINES))
    def test_the_card_states_the_format_as_the_upgrade_spells_it(self, name: str) -> None:
        mirrors = _mirrors_built_by(name)
        if not mirrors:
            pytest.skip(f"no distribution mirror under {DIST_ROOT} is built by --pipeline {name}")
        expected = 0 if name in CARDS_WITHOUT_THE_FORMAT else 1
        for manifest_path in mirrors:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            statement = card_format_statement(manifest["format"])
            for render_card in dist.PIPELINES[name].card_profiles.values():
                card = render_card(manifest, f"{HF_OWNER}/{manifest_path.parent.name}")
                assert card.count(statement) == expected, manifest_path


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


#: The README sentence that spells out the accepted set (the names usable as `--pipeline` arguments,
#: in backquotes). "13 pipeline seats across 10 families" in the same section is another sentence
#: about the **number of families**, so it is not picked up. A name may hold a period
#: (`anima-2.9b`), so only a period outside backquotes ends the sentence.
README_ACCEPTED_SET = re.compile(r"The accepted set is (?P<names>(?:`[^`]*`|[^.`])+)\.")


class TestReadme:
    """The accepted set in the README (English MUST) does not drift from the implementation.

    argparse failing on `--pipeline <name>` typed as the README says is the very first step a user
    takes giving way. By ADR 0092 decision 5 this README is also the document that declares the
    license carve-out, so the reliability of its other statements reads as the reliability of the
    carve-out.
    """

    @staticmethod
    def _accepted_names() -> list[str]:
        readme = (Path(dist.__file__).resolve().parent / "README.md").read_text(encoding="utf-8")
        match = README_ACCEPTED_SET.search(readme)
        assert match is not None, (
            "the README sentence of the accepted set not found (its wording moved)"
        )
        return re.findall(r"`([^`]+)`", match.group("names"))

    def test_the_readme_lists_every_accepted_pipeline(self) -> None:
        assert sorted(self._accepted_names()) == sorted(dist.PIPELINES)

    def test_the_readme_lists_them_in_the_help_order(self) -> None:
        """The order is the `--help` order (the default first) — so a reader can match it
        against the CLI."""
        assert self._accepted_names() == list(dist.PIPELINES)


def _torch_loaded_after(source: str) -> bool:
    """Run `source` in a fresh interpreter and return whether torch entered `sys.modules`.

    "Which modules are already imported" cannot be judged from the test session's `sys.modules`
    (other tests have already loaded torch), so it is checked every time in a subprocess, the same
    way as the core's test_package_init. The cwd is the recipe root, so `dist` / the families are
    imported under the same names as in pytest.
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
    """The distribution and card layers build without loading torch (the premise of the
    `measurements.py` of siglip2 / depth).

    Keeping the patch layer (torch-dependent) out of the import chain of the distribution path
    rests on "`import dist` does not load torch". If any one distribution / card imports the patch
    layer or a torch-dependent helper at module level, this property breaks silently.
    """

    def test_importing_the_driver_does_not_load_torch(self) -> None:
        assert not _torch_loaded_after("import dist")

    def test_importing_a_patch_layer_does_load_torch(self) -> None:
        """Control: the same check detects a torch-dependent import (the check above is not
        vacuous)."""
        assert _torch_loaded_after("import siglip2.patch")
