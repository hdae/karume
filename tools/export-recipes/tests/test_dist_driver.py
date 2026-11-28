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
import importlib
import io
import json
import os
import re
import subprocess
import sys
import urllib.error
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import pytest

import dist
from _shared.licenses import (
    APACHE_LICENSE_2_0_PATH,
    CC_BY_SA_4_0_PATH,
    FENCE_CLOSE,
    FENCE_OPEN,
    MIT_COPYRIGHT_PLACEHOLDER,
    MIT_LICENSE_PATH,
)
from _shared.paths import DIST_ROOT, INPUTS_ROOT, REPO_ROOT, SERIES_ROOT
from _shared.upstream import GEMMA_LICENSE_PAGE, UPSTREAM_AUTHORS, hf_upstreams
from anima.card import NVIDIA_NOTICE as ANIMA_NVIDIA_NOTICE
from anima.distribution import ANIMA_2_9B_PIPELINE
from anima.distribution import EXTRA_PIPELINE as ANIMA_EXTRA_PIPELINE
from anima.distribution import LICENSE_SOURCE_PATH as ANIMA_LICENSE_SOURCE_PATH
from anima.distribution import OFFICIAL_PIPELINE as ANIMA_OFFICIAL_PIPELINE
from birefnet.distribution import LUCIDA_PIPELINE
from birefnet.distribution import PIPELINE as BIREFNET_PIPELINE
from depth_anything.distribution import PIPELINE as DEPTH_ANYTHING_PIPELINE
from embeddinggemma2.distribution import PIPELINE as EMBEDDINGGEMMA2_PIPELINE
from gemma4.distribution import PIPELINE as GEMMA4_PIPELINE
from gemma4.distribution import QAT_Q4_0_PIPELINE as GEMMA4_QAT_Q4_0_PIPELINE
from gemma4.distribution import gemma4_notice_markdown
from gemma4_qat.distribution import PIPELINE as GEMMA4_QAT_PIPELINE
from irodori.distribution import GEMMA_TERMS_OF_USE_PATH
from irodori.distribution import LARGE_PIPELINE as IRODORI_LARGE_PIPELINE
from irodori.distribution import PIPELINE as IRODORI_PIPELINE
from karume.dist import (
    LICENSE_FILENAME,
    MANIFEST_FILENAME,
    NOTICE_FILENAME,
    CrossRepoReference,
    DistError,
    ExternalComponents,
    assert_name_map,
    declared_references,
    resolve_card_renderer,
    resolve_repo,
)
from karume.dist import PIPELINES as CORE_PIPELINES
from karume.migrate import card_format_statement
from karume.modelcard import (
    APACHE_LICENSE_REFERENCE,
    HF_OWNER,
    NOTICE_CLOSING,
    REVISION_PIN_COMMENT,
    RUNTIME_SENTENCE,
    UNOFFICIAL_CONVERSION_NOTICE,
    UpstreamQuote,
    hf_link,
    introduction,
)
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
#: another repository. A repository not published yet is not listed: the TS `SOURCES` it is
#: matched against get its entry at the first publication (docs/release-runbook.md §3).
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
    ("anima-2.9b", ["anima-2.9b-preview-v1"], None, "karume-anima-2.9b"),
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
    ("embeddinggemma-2", ["text"], None, "karume-embeddinggemma-2"),
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

    def test_the_six_referencing_repositories(self) -> None:
        referencing = {
            directory: sorted(
                reference.repo_name
                for reference in declared_references(dist.PIPELINES[pipeline], models)
            )
            for pipeline, models, _, directory in RELEASE_REPOSITORIES
        }
        assert {name: repos for name, repos in referencing.items() if repos} == {
            "karume-anima-extra": ["karume-anima"],
            "karume-anima-2.9b": ["karume-anima"],
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


class TestTheFormatStatementOnEveryCard:
    """Every card that names the manifest format does it once, in the one statement the upgrade of
    a repository rewrites (`karume.migrate.card_format_statement` — ADR 0153), drawn from each
    local mirror's own manifest."""

    @pytest.mark.parametrize("name", sorted(dist.PIPELINES))
    def test_the_card_states_the_format_as_the_upgrade_spells_it(self, name: str) -> None:
        mirrors = _mirrors_built_by(name)
        if not mirrors:
            pytest.skip(f"no distribution mirror under {DIST_ROOT} is built by --pipeline {name}")
        for manifest_path in mirrors:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            statement = card_format_statement(manifest["format"])
            for render_card in dist.PIPELINES[name].card_profiles.values():
                card = render_card(manifest, f"{HF_OWNER}/{manifest_path.parent.name}")
                assert card.count(statement) == 1, manifest_path


#: The published repositories, one parameter set each (`(pipeline, models, directory)`).
PUBLISHED = [
    pytest.param(pipeline, models, directory, id=directory)
    for pipeline, models, _, directory in RELEASE_REPOSITORIES
]


def _published_card(pipeline: str, directory: str) -> tuple[dict[str, Any], str]:
    """The manifest of the local mirror `models/<directory>/` and the card its pipeline draws from
    it (SKIP without the mirror, as on CI)."""
    manifest_path = DIST_ROOT / directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        pytest.skip(f"no distribution mirror {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    render_card = resolve_card_renderer(dist.PIPELINES[pipeline], None)
    return manifest, render_card(manifest, f"{HF_OWNER}/{directory}")


def _front_matter(card: str) -> dict[str, str | list[str]]:
    """The card's front matter, key → value (a list for a YAML sequence), in the card's order."""
    lines = card.split("\n")
    assert lines[0] == "---"
    fields: dict[str, str | list[str]] = {}
    key = ""
    for line in lines[1 : lines.index("---", 1)]:
        if line.startswith("  - "):
            value = fields[key]
            assert isinstance(value, list), line
            value.append(line.removeprefix("  - "))
            continue
        key, _, value_text = line.partition(": ")
        key = key.removesuffix(":")
        fields[key] = value_text if value_text else []
    return fields


def _listed(value: str | list[str]) -> list[str]:
    return value if isinstance(value, list) else [value]


def _headings(card: str) -> list[tuple[int, str]]:
    """The card's Markdown headings outside code fences, as (line index, heading line)."""
    headings = []
    fenced = False
    for index, line in enumerate(card.split("\n")):
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and re.match(r"#{1,6} ", line):
            headings.append((index, line))
    return headings


#: The front matter's keys, in the order `karume.modelcard.frontmatter` writes them.
FRONT_MATTER_KEYS = (
    "library_name",
    "pipeline_tag",
    "base_model",
    "base_model_relation",
    "license",
    "license_name",
    "license_link",
    "tags",
)

#: The three Gemma 4 repositories, whose recipes change in a stage of their own: where a rule is not
#: built for them yet, the spec names them as the exception.
GEMMA4_REPOSITORIES = frozenset({"karume-gemma4", "karume-gemma4-qat", "karume-gemma4-qat-mobile"})

#: The repositories whose `license_link` stands beside an SPDX `license`: the Gemma license page,
#: as their upstream cards link it (docs/spec/distribution-cards.md#front-matter).
GEMMA_LICENSE_LINK_REPOSITORIES = GEMMA4_REPOSITORIES | {"karume-embeddinggemma-2"}


class TestTheFrontMatterOfEveryPublishedCard:
    """docs/spec/distribution-cards.md#front-matter, on the card each published repository's
    pipeline draws from its local mirror."""

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_the_keys_come_in_the_one_order(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        _, card = _published_card(pipeline, directory)
        fields = _front_matter(card)

        assert list(fields) == [key for key in FRONT_MATTER_KEYS if key in fields]
        assert {"library_name", "pipeline_tag", "base_model", "license", "tags"} <= set(fields)
        assert fields["library_name"] == "karume"
        assert fields.get("base_model_relation", "quantized") == "quantized"

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_license_name_and_license_link_come_with_other_or_as_the_gemma_link(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        _, card = _published_card(pipeline, directory)
        fields = _front_matter(card)

        if fields["license"] == "other":
            assert "license_name" in fields and "license_link" in fields
        elif directory in GEMMA_LICENSE_LINK_REPOSITORIES:
            assert "license_name" not in fields
            assert fields["license_link"] == GEMMA_LICENSE_PAGE
        else:
            assert "license_name" not in fields and "license_link" not in fields

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_the_tags_hold_webgpu_and_begin_with_the_pipeline_tag(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        _, card = _published_card(pipeline, directory)
        fields = _front_matter(card)
        tags = _listed(fields["tags"])

        assert "webgpu" in tags
        assert "karume" not in tags
        assert tags[0] == fields["pipeline_tag"], tags


#: Words the generated texts do not use: the house word for a preset and British spellings
#: (docs/spec/distribution-cards.md#files-at-the-repository-root).
_HOUSE_AND_BRITISH_WORDS = re.compile(
    r"\b(seats?|licences?|normalis\w*|colour\w*|metres?|labelled)\b", re.IGNORECASE
)


#: The accounts whose repositories the cards name (a code-font `<owner>/<name>` of one of them is an
#: HF repository, which a card links instead).
_HF_ACCOUNTS = frozenset({*UPSTREAM_AUTHORS, HF_OWNER})

#: A URL a text writes without a link: bare, in angle brackets, or in parentheses that are not a
#: link's target.
_UNLINKED_URL = re.compile(r"<https?://|(?<![(\w/])https?://|(?<!\])\(https?://")


class TestTheLinksOfEveryPublishedRepository:
    """docs/spec/distribution-cards.md#naming-and-pinning-an-upstream — an HF repository is a
    plain-text link on a card, and no text writes a URL without a link."""

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_the_card_links_every_hf_repository_it_names_with_plain_link_text(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        _, card = _published_card(pipeline, directory)
        prose = _prose(card.split("\n---\n", 1)[1])
        coded = [
            name
            for name in re.findall(r"`([A-Za-z0-9-]+/[A-Za-z0-9._-]+)`", prose)
            # (`<family>/<major>` is a pipeline contract, not a repository)
            if name.split("/")[0] in _HF_ACCOUNTS and not name.split("/")[1].isdigit()
        ]

        assert re.findall(r"\[`[^`]+`\]\(https://huggingface\.co/", prose) == []
        assert coded == []

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_no_card_or_notice_writes_a_url_without_a_link(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        manifest, card = _published_card(pipeline, directory)
        notice = _root_files(pipeline, list(manifest["models"]))[NOTICE_FILENAME]
        # The NVIDIA notice of the Anima texts is the ruled wording of the agreement's §3.1 notice
        # and its explanation, kept as it is (docs/backlog.md).
        body = card.split("\n---\n", 1)[1].replace(ANIMA_NVIDIA_NOTICE, "")
        texts = [_prose(body), notice.replace(ANIMA_NVIDIA_NOTICE, "")]

        assert [_UNLINKED_URL.findall(text) for text in texts] == [[], []]


class TestTheWordsOfEveryPublishedRepository:
    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_the_card_and_the_notice_use_the_ordinary_terms_and_american_spelling(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        manifest, card = _published_card(pipeline, directory)
        notice = _root_files(pipeline, list(manifest["models"]))[NOTICE_FILENAME]
        # The table rows carry the mirror's preset descriptions, which change at its next build
        # (the recipes' descriptions are checked by the families' tests).
        prose = _prose("\n".join(line for line in card.split("\n") if not line.startswith("|")))

        assert _HOUSE_AND_BRITISH_WORDS.findall(prose) == []
        assert _HOUSE_AND_BRITISH_WORDS.findall(notice) == []


#: Every quotation of an upstream card the recipes write (each family's `UPSTREAM_QUOTES`).
UPSTREAM_QUOTES: list[UpstreamQuote] = [
    quote
    for family in (
        "anima",
        "birefnet",
        "depth_anything",
        "embeddinggemma2",
        "irodori",
        "moss",
        "sbv2",
    )
    for quote in importlib.import_module(f"{family}.card").UPSTREAM_QUOTES
]


def _local_upstream_card(quote: UpstreamQuote) -> Path | None:
    """The quoted README at its commit on this machine: the HF cache's snapshot, or an input
    directory whose download record (`huggingface_hub`'s `local_dir` metadata) or `REVISION.txt`
    names that commit."""
    hub = Path(
        os.environ.get("HF_HUB_CACHE")
        or Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface") / "hub"
    )
    snapshot = hub / f"models--{quote.repo.replace('/', '--')}/snapshots/{quote.revision}/README.md"
    if snapshot.is_file():
        return snapshot
    for readme in sorted(INPUTS_ROOT.glob("*/*/README.md")):
        records = (
            readme.parent / ".cache/huggingface/download/README.md.metadata",
            readme.parent / "REVISION.txt",
        )
        if any(
            record.is_file() and record.read_text().split()[:1] == [quote.revision]
            for record in records
        ):
            return readme
    return None


def _heading_text(heading: str) -> str:
    """A heading as a quotation names it: without its emphasis and leading emoji."""
    return re.sub(r"^[^\w]+", "", heading.replace("**", "")).strip()


def _sections_of(card: str, titles: frozenset[str]) -> list[tuple[frozenset[str], str]]:
    """The card's text in parts between headings, each with the headings it stands under and its
    whitespace folded (a `#` line inside a code fence is code, not a heading). A line standing
    alone whose text is one of `titles` is a section title too, at the top level: FLUX.2
    [klein]'s card titles its sections with plain lines."""
    parts: list[tuple[frozenset[str], str]] = []
    path: list[tuple[int, str]] = []
    lines: list[str] = []
    fenced = False
    rows = [*card.split("\n"), "# end"]
    for index, line in enumerate(rows):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        heading = None if fenced else re.match(r"(#{1,6})\s+(.*?)\s*#*\s*$", line)
        title = (
            not fenced
            and _heading_text(line) in titles
            and 0 < index < len(rows) - 1
            and rows[index - 1].strip() in ("", "---")
            and rows[index + 1].strip() == ""
        )
        if heading is None and not title:
            lines.append(line)
            continue
        parts.append((frozenset(text for _, text in path), " ".join(" ".join(lines).split())))
        level, text = (1, line) if heading is None else (len(heading[1]), heading[2])
        path = [*(entry for entry in path if entry[0] < level), (level, _heading_text(text))]
        lines = []
    return parts


class TestTheQuotationsOfUpstreamCards:
    """docs/spec/distribution-cards.md#what-the-upstream-says — a quotation is the upstream card's
    own words at the commit it links, under the headings it names."""

    @pytest.mark.parametrize(
        "quote",
        UPSTREAM_QUOTES,
        ids=[f"{quote.repo}:{quote.sections}" for quote in UPSTREAM_QUOTES],
    )
    def test_every_paragraph_stands_in_the_card_at_its_commit_under_the_headings_named(
        self, quote: UpstreamQuote
    ) -> None:
        readme = _local_upstream_card(quote)
        if readme is None:
            pytest.skip(f"{quote.repo} at {quote.revision} is not on this machine")
        named = frozenset(quote.sections)
        parts = _sections_of(readme.read_text(encoding="utf-8"), named)
        found: list[frozenset[str]] = []
        for paragraph in quote.paragraphs:
            under = [headings for headings, text in parts if " ".join(paragraph.split()) in text]
            assert under, f"{quote.repo}: not in the card: {paragraph[:60]!r}"
            found.extend(under)
        # Each paragraph stands under a heading named (or before the first heading when none is),
        # and each heading named holds one of them.
        assert all(headings & named or (not headings and not named) for headings in found), found
        assert named <= set().union(*found), (named, found)

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_every_blockquote_of_a_card_is_a_registered_quotation(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        if directory in GEMMA4_REPOSITORIES:
            pytest.skip("the Gemma 4 cards keep their wording until their stage")
        _, card = _published_card(pipeline, directory)
        registered = {paragraph for quote in UPSTREAM_QUOTES for paragraph in quote.paragraphs}
        quoted = [line.strip()[2:] for line in card.split("\n") if line.strip().startswith("> ")]

        assert [line for line in quoted if line not in registered] == []

    def test_every_quotation_is_written_on_a_card_with_its_source(self) -> None:
        cards = [
            _published_card(pipeline, directory)[1]
            for pipeline, _, _, directory in RELEASE_REPOSITORIES
            if directory not in GEMMA4_REPOSITORIES
        ]
        unwritten = [
            (quote.repo, quote.sections)
            for quote in UPSTREAM_QUOTES
            if not any(
                f"({quote.source}) states" in card
                and all(f"> {paragraph}\n" in card for paragraph in quote.paragraphs)
                for card in cards
            )
        ]

        assert unwritten == []


#: The card whose non-endorsement sentence also names works that are not on the Hub, by title,
#: before the `base_model` IDs
#: (docs/spec/distribution-cards.md#title-and-the-non-endorsement-sentence).
SENTENCES_WITH_OFF_HUB_WORKS = frozenset({"karume-anima-extra"})


class TestTheTitleAndTheNonEndorsementSentence:
    """docs/spec/distribution-cards.md#title-and-the-non-endorsement-sentence."""

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_the_one_level_1_heading_ends_with_karume(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        _, card = _published_card(pipeline, directory)
        titles = [line for _, line in _headings(card) if line.startswith("# ")]

        assert len(titles) == 1 and titles[0].endswith(" — Karume"), titles

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_the_sentence_names_the_base_models_and_their_publishers(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        _, card = _published_card(pipeline, directory)
        lines = card.split("\n")
        sentence = lines[
            next(index for index, line in enumerate(lines) if line.startswith("# ")) + 2
        ]
        upstreams = hf_upstreams(_listed(_front_matter(card)["base_model"]))
        expected = introduction("", upstreams)[3]

        assert (sentence == expected) is (directory not in SENTENCES_WITH_OFF_HUB_WORKS)
        named, _, publishers = sentence.partition(" for Karume. ")
        for upstream in upstreams:
            assert upstream.name in named and upstream.author in publishers


#: A shortened commit or digest of a card (the 16 hex digits and "…" a card may write for a commit
#: of another Karume repository or a file digest).
_SHORTENED_HEX = re.compile(r"`([0-9a-f]{7,39})…`")


class TestThePinsOfEveryPublishedCard:
    """docs/spec/distribution-cards.md#naming-and-pinning-an-upstream."""

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_every_base_model_is_pinned_at_its_full_commit(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        _, card = _published_card(pipeline, directory)
        unpinned = [
            repo
            for repo in _listed(_front_matter(card)["base_model"])
            if re.search(re.escape(f"{hf_link(repo)} at commit `") + "[0-9a-f]{40}`", card) is None
        ]

        assert unpinned == []

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_a_pin_is_a_commit_and_only_a_karume_commit_is_shortened(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        _, card = _published_card(pipeline, directory)

        assert "at revision `" not in card and "(revision `" not in card
        assert {len(digits) for digits in _SHORTENED_HEX.findall(card)} <= {16}


#: The level-2 sections before the family's own
#: (docs/spec/distribution-cards.md#sections-and-their-order).
STANDARD_PREAMBLE = ("What is this", "Base weights and attribution", "Models", "Usage")


class TestTheSectionsOfEveryPublishedCard:
    """docs/spec/distribution-cards.md#sections-and-their-order and
    #presets-and-the-quant-tables."""

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_the_sections_come_in_order_and_end_with_one_per_model(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        manifest, card = _published_card(pipeline, directory)
        sections = [line[3:] for _, line in _headings(card) if line.startswith("## ")]
        preamble = STANDARD_PREAMBLE
        per_model = [f"Model: {name}" for name in manifest["models"]]
        family = sections[len(preamble) : len(sections) - len(per_model)]

        assert tuple(sections[: len(preamble)]) == preamble
        assert sections[len(sections) - len(per_model) :] == per_model
        assert not set(family) & {*preamble, *per_model}, family

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_every_model_section_opens_with_its_quant_table(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        manifest, card = _published_card(pipeline, directory)
        headings = [line for _, line in _headings(card)]

        for name in manifest["models"]:
            assert headings[headings.index(f"## Model: {name}") + 1] == "### Quants", name
        assert card.count("| Model | Pipeline | Quants | Default quant |") == 1
        assert card.count("| Quant | What it is | Download | Weights | Compute |") == len(
            manifest["models"]
        )


class TestTheRuntimeSentenceOfEveryPublishedCard:
    """docs/spec/distribution-cards.md#what-is-this — the browser, Node and Deno in one sentence,
    and the reader the container needs."""

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_what_is_this_says_which_reader_the_container_needs(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        _, card = _published_card(pipeline, directory)
        lines = card.split("\n")
        start = lines.index("## What is this")
        end = next(
            index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")
        )
        section = " ".join(" ".join(lines[start:end]).split())

        assert re.search(
            r"- Not readable by .+; the reader is a pipeline that implements `[^`]+`\.", section
        )

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_what_is_this_carries_the_runtime_sentence_once_as_a_line(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        _, card = _published_card(pipeline, directory)
        lines = card.split("\n")
        start = lines.index("## What is this")
        end = next(
            index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")
        )
        assert lines.count(RUNTIME_SENTENCE) == 1
        assert start < lines.index(RUNTIME_SENTENCE) < end


#: The bullet that sends the reader to the upstream cards (`karume.modelcard.upstream_cards`).
_UPSTREAM_CARDS = "- **What the model can and cannot do** — "

#: The cards whose upstream-card bullet does not list every `base_model` yet, or does not end the
#: attribution section: the Gemma 4 ones, whose recipes change in a stage of their own
#: (docs/spec/distribution-cards.md#what-the-upstream-says).
CARDS_WITH_A_PARTIAL_UPSTREAM_CARD_BULLET = GEMMA4_REPOSITORIES


class TestTheUpstreamCardsOfEveryPublishedCard:
    """docs/spec/distribution-cards.md#what-the-upstream-says."""

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_the_attribution_ends_by_sending_the_reader_to_every_upstream_card(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        _, card = _published_card(pipeline, directory)
        lines = card.split("\n")
        bullets = [line for line in lines if line.startswith(_UPSTREAM_CARDS)]
        attribution = lines[: lines.index("## Models")]
        last = next(line for line in reversed(attribution) if line)
        listed = all(hf_link(repo) in last for repo in _listed(_front_matter(card)["base_model"]))

        assert len(bullets) <= 1
        complete = bullets == [last] and listed
        assert complete is (directory not in CARDS_WITH_A_PARTIAL_UPSTREAM_CARD_BULLET), bullets


class TestTheUsageOfEveryPublishedCard:
    """docs/spec/distribution-cards.md#usage."""

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_the_usage_opens_with_a_pinnable_from_pretrained_of_this_repository(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        _, card = _published_card(pipeline, directory)
        lines = card.split("\n")
        pin = "\n".join((f'  repo: "{HF_OWNER}/{directory}",', *REVISION_PIN_COMMENT))

        body = lines[lines.index("## Usage") + 1 :]
        block = body[body.index("```ts") : body.index("```", body.index("```ts") + 1)]
        assert next(line for line in body if line) == "```ts"
        assert any(
            re.match(r'import .* from "jsr:@karume/models/[a-z0-9-]+";$', line) for line in block
        )
        assert re.search(
            r"^(await )?using pipeline = await \w+\.fromPretrained\(\{$", "\n".join(block), re.M
        )
        assert card.count(pin) == 1
        assert pin in "\n".join(block)


#: The head of every card's changes bullet (`karume.modelcard.changes_made_here`).
_CHANGES_HEAD = re.compile(
    re.escape("- **Changes made here** (listed in full in `NOTICE.md`") + r"(, per [^:]+)?\):"
)


class TestTheLegalPointersOfEveryPublishedCard:
    """docs/spec/distribution-cards.md#license-statements."""

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_the_card_names_license_md_and_notice_md(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        _, card = _published_card(pipeline, directory)

        assert "`LICENSE.md`" in card and "`NOTICE.md`" in card

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_one_changes_bullet_points_at_the_notice_listing_them_in_full(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        _, card = _published_card(pipeline, directory)
        bullets = [line for line in card.split("\n") if line.startswith("- **Changes made here**")]

        assert len(bullets) == 1
        assert _CHANGES_HEAD.match(bullets[0]) is not None, bullets


def _root_files(pipeline: str, models: list[str]) -> Mapping[str, str]:
    """The legal texts the driver hands the core for a build of `models` (`dist.ROOT_FILES` for a
    pipeline whose texts follow the models, else the pipeline's fixed set)."""
    root_files = dist.ROOT_FILES.get(pipeline)
    return root_files(models) if root_files is not None else dist.PIPELINES[pipeline].root_files


def _prose(card: str) -> str:
    """The card's text outside code fences and blockquotes (what the recipe says in its own words),
    runs of whitespace as one space."""
    lines = []
    fenced = False
    for line in card.split("\n"):
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and not line.lstrip().startswith(">"):
            lines.append(line)
    return " ".join(" ".join(lines).split())


#: "licensed X" in a card's own words, but for "licensed by" (the licensor of a notice).
_LICENSED = re.compile(r"\blicensed (?!by\b)(\S+)")

#: An HF license identifier in bold, as a card writes it (`**apache-2.0**`), with its punctuation.
_BOLD_LICENSE_ID = re.compile(r"\*\*[a-z0-9][a-z0-9.-]*\*\*[,.;:)]*")

#: What a notice does not call a license: its short names (the notice names a license by its full
#: title — docs/spec/distribution-cards.md#noticemd).
_SHORT_LICENSE_NAMES = (
    "Apache 2.0",
    "Apache-2.0",
    "Apache License 2.0",
    "CC BY-SA 4.0",
    "(MIT)",
    "CC-BY",
)


class TestTheLicenseNamesOfEveryPublishedRepository:
    """docs/spec/distribution-cards.md#license-statements — a card names a license by its HF
    identifier in bold, a notice by its full title."""

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_every_license_a_card_states_is_its_hf_identifier_in_bold(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        _, card = _published_card(pipeline, directory)
        named = [match.group(1) for match in _LICENSED.finditer(_prose(card))]
        other = [name for name in named if not _BOLD_LICENSE_ID.fullmatch(name)]

        assert named
        assert other == []

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_an_apache_card_points_at_the_text_in_the_one_wording(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        """The weights whose license is `LICENSE.md`'s primary Apache text are followed by
        `APACHE_LICENSE_REFERENCE`, and no card spells the full-text link another way."""
        _, card = _published_card(pipeline, directory)
        apache = PRIMARY_LICENSES[directory] == _APACHE

        assert (f"licensed **apache-2.0** {APACHE_LICENSE_REFERENCE}." in card) is apache
        assert card.count("[full text](") == card.count(APACHE_LICENSE_REFERENCE)

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_a_notice_calls_no_license_by_a_short_name(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        notice = _root_files(pipeline, models)[NOTICE_FILENAME]

        assert [name for name in _SHORT_LICENSE_NAMES if name in notice] == []


#: The notices that do not name every `base_model` of their card by its repository ID: none
#: (docs/spec/distribution-cards.md#noticemd).
NOTICES_WITHOUT_EVERY_UPSTREAM_ID: frozenset[str] = frozenset()


class TestTheUpstreamsEveryNoticeNames:
    """docs/spec/distribution-cards.md#noticemd — a notice names each upstream by its repository ID
    in code font."""

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_the_notice_names_every_base_model_of_its_card_in_code_font(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        manifest, card = _published_card(pipeline, directory)
        # The notice of the models the mirror holds (the card is drawn from the same manifest).
        notice = _root_files(pipeline, list(manifest["models"]))[NOTICE_FILENAME]
        unnamed = [
            repo for repo in _listed(_front_matter(card)["base_model"]) if f"`{repo}`" not in notice
        ]

        assert "listed in `README.md`" not in notice
        assert bool(unnamed) is (directory in NOTICES_WITHOUT_EVERY_UPSTREAM_ID), unnamed


_APACHE = ("apache", "the Apache License, Version 2.0")
_MIT = ("mit", "the MIT License")
_CIRCLESTONE = ("circlestone", "the CircleStone Labs Non-Commercial License v1.2")

#: Repository → the primary license text `LICENSE.md` begins with, and the title its `NOTICE.md`
#: names it by (docs/spec/distribution-cards.md#licensemd and #noticemd).
PRIMARY_LICENSES = {
    "karume-anima": _CIRCLESTONE,
    "karume-anima-extra": _CIRCLESTONE,
    "karume-anima-2.9b": _CIRCLESTONE,
    "karume-birefnet-hr": _MIT,
    "karume-lucida": _MIT,
    "karume-depth-anything-v2": _APACHE,
    "karume-gemma4": _APACHE,
    "karume-gemma4-qat": _APACHE,
    "karume-gemma4-qat-mobile": _APACHE,
    "karume-irodori-v4-small": _MIT,
    "karume-irodori-v4.1-small": _MIT,
    "karume-irodori-v4.1-small-mf": _MIT,
    "karume-irodori-v4-large": ("gemma-terms", "the Gemma Terms of Use"),
    "karume-sbv2-jvnv": (
        "cc-by-sa",
        "Creative Commons Attribution-ShareAlike 4.0 International",
    ),
    "karume-siglip2": _APACHE,
    "karume-umt5-xxl": _APACHE,
    "karume-wan2.2": _APACHE,
    "karume-moss-image": _APACHE,
    "karume-embeddinggemma-2": _APACHE,
}

#: The verbatim texts other than MIT's (whose copyright lines are inserted).
_VERBATIM_TEXTS = {
    "apache": APACHE_LICENSE_2_0_PATH,
    "cc-by-sa": CC_BY_SA_4_0_PATH,
    "circlestone": ANIMA_LICENSE_SOURCE_PATH,
    "gemma-terms": GEMMA_TERMS_OF_USE_PATH,
}


def _normalized(text: str) -> str:
    """Runs of whitespace as one space (a sentence may be wrapped anywhere)."""
    return " ".join(text.split())


class TestTheNoticeOfEveryPublishedRepository:
    """docs/spec/distribution-cards.md#noticemd, on the notice the driver writes for each published
    repository (no mirror needed)."""

    def test_every_published_repository_has_its_primary_license_stated(self) -> None:
        assert set(PRIMARY_LICENSES) == {directory for *_, directory in RELEASE_REPOSITORIES}

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_the_notice_names_the_license_then_the_changes_then_the_closing(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        notice = _root_files(pipeline, models)[NOTICE_FILENAME]
        text = _normalized(notice)

        _, title = PRIMARY_LICENSES[directory]
        changes = text.index("The following changes were made:")
        assert notice.startswith("# NOTICE\n\n")
        assert title is not None and title in text[:changes]
        assert text.count("The following changes were made:") == 1
        assert text.count("No retraining and no fine-tuning were performed.") == 1
        assert text.index("No retraining and no fine-tuning were performed.") > changes

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_the_closing_paragraph_is_the_one_wording_on_one_line(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        """`karume.modelcard.notice_closing`: the shared sentence, then what is not distributed."""
        lines = _root_files(pipeline, models)[NOTICE_FILENAME].split("\n")
        closing = [line for line in lines if line.startswith(NOTICE_CLOSING)]

        assert len(closing) == 1
        assert closing[0].endswith(" not distributed here.")


#: The repositories whose `LICENSE.md` begins with the primary text unfenced: the Gemma 4 ones,
#: whose recipes change in a stage of their own (docs/spec/distribution-cards.md#licensemd).
UNFENCED_LICENSES = GEMMA4_REPOSITORIES

#: The heading of a further license text: `## <license> — <scope>`.
_FURTHER_HEADING = re.compile(r"## \S.* — \S.*")


class TestTheLicenseTextOfEveryPublishedRepository:
    """docs/spec/distribution-cards.md#licensemd — the primary license text fenced from the first
    byte and verbatim inside the fence; every further text after a `---` line, its heading and
    its paragraph, fenced the same way."""

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_license_md_begins_with_the_primary_text_byte_for_byte(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        license_text = _root_files(pipeline, models)[LICENSE_FILENAME]
        primary, _ = PRIMARY_LICENSES[directory]
        fence = "" if directory in UNFENCED_LICENSES else FENCE_OPEN

        assert license_text.startswith(fence)
        body = license_text[len(fence) :]
        if primary != "mit":
            text = _VERBATIM_TEXTS[primary].read_text(encoding="utf-8")
            closing = "" if not fence else ("" if text.endswith("\n") else "\n") + FENCE_CLOSE
            assert body.startswith(text + closing)
            return
        head, tail = MIT_LICENSE_PATH.read_text(encoding="utf-8").split(MIT_COPYRIGHT_PLACEHOLDER)
        assert body.startswith(head)
        rest = body[len(head) :]
        # The template after the placeholder follows the copyright lines verbatim.
        copyright_end = rest.find(tail)
        assert copyright_end > 0
        assert all(line.startswith("Copyright (c) ") for line in rest[:copyright_end].split("\n"))
        assert rest[copyright_end + len(tail) :].startswith(FENCE_CLOSE if fence else "")

    @pytest.mark.parametrize(("pipeline", "models", "directory"), PUBLISHED)
    def test_every_further_text_follows_a_rule_a_scoped_heading_and_a_paragraph_in_a_fence(
        self, pipeline: str, models: list[str], directory: str
    ) -> None:
        if directory in UNFENCED_LICENSES:
            pytest.skip("the Gemma 4 LICENSE.md is the Apache text alone, unfenced")
        license_text = _root_files(pipeline, models)[LICENSE_FILENAME]
        # A further text follows the closing fence of the one before it.
        _, *further = re.split(r"(?<=```\n)\n---\n\n", license_text)

        for section in further:
            heading, paragraph, rest = section.split("\n\n", 2)
            assert _FURTHER_HEADING.fullmatch(heading), heading
            assert paragraph and not paragraph.startswith(("#", "`"))
            assert rest.startswith(FENCE_OPEN) and rest.endswith(FENCE_CLOSE)
            assert rest.count("```") == 2


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


class TestRootFilesOfTheModelsBuilt:
    """The root files the driver hands the core are those of the models built, for a pipeline whose
    legal texts depend on them ({@link dist.ROOT_FILES}); the core writes them as given."""

    @staticmethod
    def _handed_to_core(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pipeline: str, models: list[str]
    ) -> dict[str, str]:
        handed: dict[str, Any] = {}

        def core(arguments: list[str], **kwargs: Any) -> None:
            handed["pipelines"] = kwargs["pipelines"]

        monkeypatch.setattr(dist, "dist_main", core)
        dist.main(
            [
                "--pipeline",
                pipeline,
                *(argument for model in models for argument in ("--model", model)),
                "--out",
                str(tmp_path / "out"),
            ],
            get=_no_hub,
        )
        return dict(handed["pipelines"][pipeline].root_files)

    def test_a_gemma4_build_of_e2b_alone_names_the_e2b_drafter_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        notice = self._handed_to_core(monkeypatch, tmp_path, "gemma4", ["e2b"])["NOTICE.md"]

        assert "google/gemma-4-E2B-it-assistant" in notice
        assert "google/gemma-4-E4B-it-assistant" not in notice
        assert notice == gemma4_notice_markdown(["e2b"])

    def test_a_gemma4_build_of_both_models_names_both_drafters(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        notice = self._handed_to_core(monkeypatch, tmp_path, "gemma4", ["e2b", "e4b"])["NOTICE.md"]

        assert "google/gemma-4-E2B-it-assistant" in notice
        assert "google/gemma-4-E4B-it-assistant" in notice
        assert notice == gemma4_notice_markdown(["e2b", "e4b"])

    def test_a_pipeline_without_an_entry_keeps_its_fixed_root_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        assert "depth-anything" not in dist.ROOT_FILES

        root_files = self._handed_to_core(
            monkeypatch, tmp_path, "depth-anything", [DEPTH_ANYTHING_PIPELINE.default_model]
        )

        assert root_files == dict(DEPTH_ANYTHING_PIPELINE.root_files)

    def test_a_siglip2_build_of_one_model_names_its_checkpoint_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        notice = self._handed_to_core(monkeypatch, tmp_path, "siglip2", ["base"])["NOTICE.md"]

        assert "`google/siglip2-base-patch16-224`" in notice
        assert "google/siglip2-so400m-patch14-384" not in notice


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
