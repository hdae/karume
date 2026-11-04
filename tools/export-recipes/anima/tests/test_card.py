"""Anima's model card rendering (`anima.card`) — the two cards of the official repository and the
fine-tune repository.

The real `models/karume-anima/karume.json` is not used — the job here is to see that the values
of a fake manifest show up in the text as they are (= not a single hand-written number is mixed
in), so building it with values **different** from the real ones detects more. The numbers, paths
and presentation fields stay fake values that do not overlap the real ones; **only the model
names** use the real spelling (the keys of {@link UPSTREAM_MODELS}) — the origins section looks
up the attribution from the manifest's model names, so with made-up names neither card can be
rendered at all.

Since manifest v2 (`karume/2` — ADR 0041), a card describes several models of one repository.
The fake manifest defaults to the family shape (2 models, with shared files) so that rendering
that works only with a single model does not slip through.

There have been two cards since the restructuring of 2026-09-01 (ADR 0087) — the official
repository (CircleStone's 5 variants together, default = Turbo) and the fine-tune repository
(third-party fine-tunes) differ in both the license notice and the introduction of the origins
section. Turbo became an official checkpoint, so **the attribution section for a baked-in LoRA is
gone entirely** (the old turbo-only card and its attribution check were removed in that
iteration).
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from anima.card import (
    ANIMA_EXTRA_TITLE,
    ANIMA_OFFICIAL_TITLE,
    ATTRIBUTION_NOTICE,
    EXTRA_ORIGINS_INTRO,
    NVIDIA_NOTICE,
    OFFICIAL_ORIGINS_INTRO,
    RIGHTS_GRANT_STATEMENT,
    SUPPORTED_PIPELINE,
    UPSTREAM_MODELS,
    render_base_card,
    render_extra_card,
)
from anima.distribution import ANIMA_QUANT_ABBREVIATIONS, LICENSE_SOURCE_PATH

#: The repository ID spelled into the usage snippet (dist passes it from the pipeline's
#: declaration).
REPO = "hdae/fake-repo"

#: The models the official repository's fixture lists (default = Turbo = the guidance 1
#: entry).
OFFICIAL_DEFAULT = "anima-turbo-v1.1"
OFFICIAL_SECOND = "anima-aesthetic-v1.1"

#: The models the fine-tune repository's fixture lists (the 2nd is a source whose
#: `allowDifferentLicense` is false — the one that gets the no-relicensing note).
EXTRA_DEFAULT = "anima-wai-v1.0"
EXTRA_SECOND = "anima-copycat-20260610"


def _ref(path: str, size: int, digit: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digit * 64}


def _container(*refs: dict[str, Any]) -> dict[str, Any]:
    """One dtype of weights (the `container` of `karume/5` — ADR 0109 decision 3).

    The card does not read `descriptor`, so only its shape follows the real one (the lengths and
    sha256 of the 2 documents).
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


def _numbered(model: str, label: str, index: int, total: int) -> str:
    """The relative path of one transformer part (the numbering convention is container-v1
    §8)."""
    return f"{model}/transformer/model.{label}-{index:05d}-of-{total:05d}.krm"


def _two_model_manifest(default_name: str, second_name: str) -> dict[str, Any]:
    """The minimal manifest that keeps the shape of ADR 0041 (fake values that do not overlap the
    real ones).

    2 models sharing the text_encoder under `shared/`. The default model is guidance 1 (= the
    negative prompt has no effect), the 2nd is on the CFG side, so one card goes through both
    branches.
    """
    shared_encoder = _ref("shared/text_encoder/model.krm", 111, "a")
    return {
        "format": "karume/5",
        "generator": "karume/9.9.9",
        "defaultModel": default_name,
        "models": {
            default_name: {
                "pipeline": SUPPORTED_PIPELINE,
                "weights": {
                    "text_encoder": {"f16": _container(shared_encoder)},
                    # The rope base table is an **asset** of the container (ADR 0109 decision
                    # 4) — it sits in the part list as a dedicated part, so each entry gains one
                    # 333 B part.
                    "transformer": {
                        "f16": _container(
                            _ref(_numbered(default_name, "f16", 1, 2), 222, "b"),
                            _ref(_numbered(default_name, "f16", 2, 2), 333, "c"),
                        ),
                        "i8": _container(
                            _ref(_numbered(default_name, "i8", 1, 2), 444, "d"),
                            _ref(_numbered(default_name, "i8", 2, 2), 333, "9"),
                        ),
                    },
                },
                "assets": {"tokenizer": _ref("shared/tokenizer/qwen2.json", 555, "e")},
                "quants": {
                    "f16": {
                        "weights": {"text_encoder": "f16", "transformer": "f16"},
                        "session": {},
                    },
                    "f16+dit8-a8": {
                        "weights": {"text_encoder": "f16", "transformer": "i8"},
                        "session": {"linearCompute": "a8"},
                        "label": "fake label",
                        "description": "Fake description.",
                    },
                    "f16-c16": {
                        "weights": {"text_encoder": "f16", "transformer": "f16"},
                        "session": {"linearCompute": "f16"},
                        "gpuFeatures": {"shaderF16": True},
                    },
                },
                "defaultQuant": "f16+dit8-a8",
                "pipelineConfig": {
                    "scheduler": {"shift": 3, "numTrainTimesteps": 1000},
                    "defaults": {
                        "steps": 7,
                        "guidanceScale": 1,
                        "resolution": {"width": 640, "height": 384},
                        "negativePrompt": "fake negative value",
                    },
                },
            },
            second_name: {
                "pipeline": SUPPORTED_PIPELINE,
                "weights": {
                    "text_encoder": {"f16": _container(shared_encoder)},
                    "transformer": {
                        "i8": _container(_ref(f"{second_name}/transformer/model.i8.krm", 666, "f"))
                    },
                },
                "assets": {"tokenizer": _ref("shared/tokenizer/qwen2.json", 555, "e")},
                "quants": {
                    "f16+dit8": {
                        "weights": {"text_encoder": "f16", "transformer": "i8"},
                        "session": {},
                    }
                },
                "defaultQuant": "f16+dit8",
                "pipelineConfig": {
                    "scheduler": {"shift": 3, "numTrainTimesteps": 1000},
                    "defaults": {
                        "steps": 4,
                        "guidanceScale": 2,
                        "resolution": {"width": 128, "height": 256},
                        "negativePrompt": "another negative",
                    },
                },
            },
        },
    }


def _official_manifest() -> dict[str, Any]:
    """The shape of the official repository (karume-anima) — default = Turbo (guidance 1) + the
    official Aesthetic."""
    return _two_model_manifest(OFFICIAL_DEFAULT, OFFICIAL_SECOND)


def _extra_manifest() -> dict[str, Any]:
    """The shape of the fine-tune repository (karume-anima-extra) — its default is on the CFG side
    too (many steps + guidance)."""
    manifest = _two_model_manifest(EXTRA_DEFAULT, EXTRA_SECOND)
    defaults = manifest["models"][EXTRA_DEFAULT]["pipelineConfig"]["defaults"]
    defaults["guidanceScale"] = 4
    defaults["steps"] = 28
    return manifest


def _official(manifest: dict[str, Any]) -> str:
    return render_base_card(manifest, REPO, ANIMA_QUANT_ABBREVIATIONS)


def _extra(manifest: dict[str, Any]) -> str:
    return render_extra_card(manifest, REPO, ANIMA_QUANT_ABBREVIATIONS)


@pytest.fixture
def card() -> str:
    return _official(_official_manifest())


@pytest.fixture
def extra_card() -> str:
    return _extra(_extra_manifest())


def _quant_row(card: str, name: str) -> str:
    """Pick that entry's row from the quant table (by entry name — independent of column positions
    and of the default mark)."""
    rows = [line for line in card.splitlines() if line.startswith(f"| `{name}`")]
    assert len(rows) == 1, name
    return rows[0]


def _split_manifest() -> dict[str, Any]:
    """A distribution whose weights are split into **even more parts** (splitting above 1GiB —
    container-v1 §8).

    The real anima distribution has this shape (the transformer in 2–4 parts + the asset part).
    With only fixtures whose weights are a single part, prose saying "container = 1 file" would
    slip through while contradicting the facts (it actually happened on a published card —
    X2-103).
    """
    manifest = _official_manifest()
    manifest["models"][OFFICIAL_DEFAULT]["weights"]["transformer"]["f16"]["container"]["parts"] = [
        _ref(_numbered(OFFICIAL_DEFAULT, "f16", 1, 3), 777, "1"),
        _ref(_numbered(OFFICIAL_DEFAULT, "f16", 2, 3), 888, "2"),
        _ref(_numbered(OFFICIAL_DEFAULT, "f16", 3, 3), 333, "c"),
    ]
    return manifest


@pytest.fixture
def split_card() -> str:
    return _official(_split_manifest())


class TestSplitDistribution:
    """With a split distribution (the real distribution's shape), the card does not contradict the
    facts."""

    def test_it_counts_every_part_of_a_split_component(self, card: str, split_card: str) -> None:
        """The Download column adds up every single part (part 0 alone would understate the download
        size).

        Before the split the f16 entry is 111 + 222 + rope part 333 + tokenizer 555 = 1,221 B
        (= 1.19 KiB); after it, 222 splits into 777 + 888 for 2,664 B (= 2.60 KiB).
        """
        assert "| 1.19 KiB (" in _quant_row(card, "f16")
        assert "| 2.60 KiB (" in _quant_row(split_card, "f16")

    def test_it_never_calls_the_container_a_single_file(
        self, card: str, split_card: str, extra_card: str
    ) -> None:
        """One wording regardless of splitting (no manifest-dependent variants — X2-103 ruling a).

        The list of modifications in `NOTICE.md` is fixed when the Pipeline is built and cannot
        see the manifest, so varying only the card would make the two documents say different
        things. Hence **no manifest** gets "a single safetensors file".
        """
        for text in (card, split_card, extra_card):
            flat = " ".join(text.split())
            assert "a single safetensors file" not in flat
            assert "`.krm` part sequence" in flat

    def test_it_does_not_advertise_the_local_asset_entry_point(
        self, card: str, split_card: str, extra_card: str
    ) -> None:
        """`fromAssets` is not advertised (ruling of 2026-08-29) — the entry point for using it from
        HF is `fromPretrained`."""
        for text in (card, split_card, extra_card):
            assert "fromAssets" not in text


class TestOfficialCard:
    """The official repository's card (`render_base_card`) — the card of the repository holding
    CircleStone's 5 variants."""

    def test_it_titles_the_repository_as_the_official_one(self, card: str) -> None:
        assert f"# {ANIMA_OFFICIAL_TITLE}" in card
        assert f"# {ANIMA_EXTRA_TITLE}" not in card

    def test_it_introduces_every_model_as_an_official_release(self, card: str) -> None:
        """The introduction of the origins section is a fact that differs per repository — mixing
        them up swaps the attribution claim itself."""
        for line in OFFICIAL_ORIGINS_INTRO:
            assert line in card
        assert EXTRA_ORIGINS_INTRO[0] not in card

    def test_it_never_mentions_a_baked_lora(self, card: str) -> None:
        """Turbo became an official checkpoint, and the baked-in LoRA is gone (2026-09-01).

        If the attribution section remained, a distribution with nothing baked in would claim
        "this LoRA was baked in" — the prose is plausible as values go, so a reader would notice
        the contradiction only by tracing upstream. The check covers **the whole text** because
        there was a real case where, with the attribution section removed, one line of prose
        kept talking about baking (the step explanation of `_usage` still said "the baked-in LoRA
        is distilled for…").
        """
        assert "LoRA" not in card
        assert "## Baked-in LoRA" not in card
        assert "civitai.com/models/2560840" not in card
        assert "1b55e40bdb1d0e5a78cb498f245fccfdaae97823265db957d2aabdcf4cd3caf1" not in card

    def test_it_shows_the_attribution_notice_verbatim(self, card: str) -> None:
        """The display requirement of §3(b) is met verbatim (a summary is not a display)."""
        assert ATTRIBUTION_NOTICE in card

    def test_it_specifies_that_the_rights_are_granted_directly_by_the_company(
        self, card: str
    ) -> None:
        """The explicit statement of the latter half of §3(a) carries, verbatim, the same single
        constant as `NOTICE.md`."""
        assert RIGHTS_GRANT_STATEMENT in card

    def test_it_carries_the_nvidia_open_model_license_part(self, card: str) -> None:
        """NVIDIA Open Model License §3.1 / §3.2 — the same constant as `NOTICE.md`, inside the
        License section, with the verbatim attribution notice on its own line and "Built on
        NVIDIA Cosmos" (§3.2 asks for it in the product documentation)."""
        _, _, license_section = card.partition("## License")
        section, _, _ = license_section.partition("## Models")
        lines = section.split("\n")
        assert NVIDIA_NOTICE in section
        assert "Licensed by NVIDIA Corporation under the NVIDIA Open Model License" in lines
        assert "Built on NVIDIA Cosmos." in lines

    def test_the_license_section_says_license_md_holds_both_texts(self, card: str) -> None:
        """`LICENSE.md` is no longer one text: the card must not call it "the full license text"."""
        _, _, license_section = card.partition("## License")
        section, _, _ = license_section.partition("## Models")
        flat = " ".join(section.split())
        assert "(the full license text)" not in flat
        assert (
            "`LICENSE.md` (the full text of that license, followed by the NVIDIA Open Model"
            " License Agreement)" in flat
        )

    def test_it_states_that_the_outputs_stay_the_users_own(self, card: str) -> None:
        """License v1.2 §2(e) — the non-commercial restriction applies to the weights, not to the
        outputs.

        It is the first thing a reader wants to know, so the License section names it (checked
        against the primary source 2026-09-01).
        """
        _, _, license_section = card.partition("## License")
        section, _, _ = license_section.partition("## Models")
        flat = " ".join(section.split())
        assert (
            "Outputs you generate are yours to use for any purpose, including commercially" in flat
        )
        assert "the non-commercial restriction applies to the model weights and derivatives" in flat

    def test_the_license_section_points_at_the_two_files_shipped_alongside(self, card: str) -> None:
        """The license text of §3(a) and the Notice of §3(d) are shipped alongside — the card says
        where they are."""
        _, _, license_section = card.partition("## License")
        section, _, _ = license_section.partition("## Models")
        assert "`LICENSE.md`" in section
        assert "`NOTICE.md`" in section

    def test_the_license_section_names_the_bundled_version(self, card: str) -> None:
        """The section names the version of the text shipped as `LICENSE.md` (v1.2 — ruling 37 a,
        2026-10-29); the §2(e) it cites for Outputs is the v1.2 numbering."""
        _, _, license_section = card.partition("## License")
        section, _, _ = license_section.partition("## Models")
        flat = " ".join(section.split())
        title = LICENSE_SOURCE_PATH.read_text(encoding="utf-8").splitlines()[0]
        assert title == "CircleStone Labs Non-Commercial License v1.2"
        assert f"{title} (non-commercial use only)" in flat

    def test_the_license_section_denies_any_official_standing(self, card: str) -> None:
        """§3(d)(iii) — must not be mistaken for an official or endorsed product (checked as **a
        sentence**, regardless of where lines wrap)."""
        _, _, license_section = card.partition("## License")
        section, _, _ = license_section.partition("## Models")
        assert (
            "not an official product of CircleStone Labs LLC, and it is not endorsed, approved or"
            " validated by CircleStone Labs LLC." in " ".join(section.split())
        )

    def test_it_stays_silent_about_third_party_permissions(self, card: str) -> None:
        """Only CircleStone's single license applies to the official repository.

        If the notes meant for third-party fine-tunes (the source page's permissions, no
        relicensing) always appeared, readers would take it that other conditions apply to this
        repository too.
        """
        assert "do not relicense it" not in card
        assert "community fine-tunes are redistributed" not in " ".join(card.split())

    def test_it_lists_the_origin_of_every_model_in_the_manifest(self, card: str) -> None:
        for name in (OFFICIAL_DEFAULT, OFFICIAL_SECOND):
            upstream = UPSTREAM_MODELS[name]
            assert f"### `{name}` — {upstream.title}" in card
            assert upstream.author in card
            assert upstream.source in card
            assert f"- **Converted from**: `{upstream.file}`" in card

    def test_it_lists_the_origins_in_manifest_order(self, card: str) -> None:
        """The origins section keeps the manifest's order — the same order as the `models` table and
        the per-model sections.

        Sorting by name would bring to the front what should come after the default model (this
        really happened: `anima-copycat-…` came first and was pointed out). Two orders mixed in
        one card make readers think "this order means something".
        """
        order = list(_official_manifest()["models"])
        assert order != sorted(order), (
            "an order that equals name order cannot test the ordering claim"
        )
        positions = [card.index(f"### `{name}` — ") for name in order]
        assert positions == sorted(positions)

    def test_it_refuses_a_model_whose_origin_is_unknown(self) -> None:
        """A model not in the origins table **cannot get an attribution** — it fails instead of
        being silently omitted."""
        manifest = _official_manifest()
        manifest["models"]["anima-nope"] = manifest["models"][OFFICIAL_DEFAULT]

        with pytest.raises(ValueError, match=r"has no origin in card\.py"):
            _official(manifest)

    def test_it_refuses_another_pipelines_manifest(self) -> None:
        manifest = _official_manifest()
        manifest["models"][OFFICIAL_DEFAULT]["pipeline"] = "sbv2/1"

        with pytest.raises(ValueError):
            _official(manifest)


class TestExtraCard:
    """The fine-tune repository's card (`render_extra_card`) — the card of the repository that
    lists only third-party fine-tunes."""

    def test_it_titles_the_repository_as_the_community_one(self, extra_card: str) -> None:
        assert f"# {ANIMA_EXTRA_TITLE}" in extra_card
        assert f"# {ANIMA_OFFICIAL_TITLE}" not in extra_card

    def test_it_introduces_every_model_as_a_community_fine_tune(self, extra_card: str) -> None:
        for line in EXTRA_ORIGINS_INTRO:
            assert line in extra_card
        assert OFFICIAL_ORIGINS_INTRO[0] not in extra_card

    def test_it_points_the_reader_at_the_official_repository(self, extra_card: str) -> None:
        """The text stack is held as a cross-repository reference to the official repository (the
        pinned reference of `karume.json`).

        Without saying where the weights come from, readers take it that "this repository is
        self-contained".
        """
        assert "hdae/karume-anima" in extra_card
        assert "The official models (base / Aesthetic /" in extra_card

    def test_it_never_mentions_a_baked_lora(self, extra_card: str) -> None:
        """Naming a LoRA in a distribution with no LoRA baked in makes the attribution itself a
        lie."""
        assert "LoRA" not in extra_card

    def test_it_shows_the_attribution_notice_verbatim(self, extra_card: str) -> None:
        """The display requirement of §3(b) is met verbatim, as on the official repository's
        card."""
        assert ATTRIBUTION_NOTICE in extra_card

    def test_it_specifies_that_the_rights_are_granted_directly_by_the_company(
        self, extra_card: str
    ) -> None:
        """The explicit statement of the latter half of §3(a) carries, verbatim, the same single
        constant as the official repository's card."""
        assert RIGHTS_GRANT_STATEMENT in extra_card

    def test_it_carries_the_nvidia_open_model_license_part(self, extra_card: str) -> None:
        """NVIDIA Open Model License §3.1 / §3.2 — the same constant as the official card."""
        lines = extra_card.split("\n")
        assert NVIDIA_NOTICE in extra_card
        assert "Licensed by NVIDIA Corporation under the NVIDIA Open Model License" in lines
        assert "Built on NVIDIA Cosmos." in lines

    def test_it_lists_the_permissions_of_every_source_page(self, extra_card: str) -> None:
        """The permissions of the source page differ per author — carried **verbatim per
        source**."""
        for name in (EXTRA_DEFAULT, EXTRA_SECOND):
            for key, value in UPSTREAM_MODELS[name].permissions:
                assert f"- `{key}`: {value}" in extra_card

    def test_it_warns_when_a_source_forbids_relicensing(self, extra_card: str) -> None:
        """A source whose `allowDifferentLicense` is false requires "redistribute under the same
        terms"."""
        assert "allowDifferentLicense`: false" in extra_card
        assert "do not relicense it" in extra_card
        assert f"`{EXTRA_SECOND}`" in extra_card.partition("do not relicense it")[0]

    def test_it_stays_silent_about_relicensing_when_no_source_forbids_it(self) -> None:
        """A note that matters only for some repositories would, if always shown, make readers
        misread the terms."""
        manifest = _extra_manifest()
        del manifest["models"][EXTRA_SECOND]

        assert "do not relicense it" not in _extra(manifest)

    def test_it_never_lists_a_model_the_manifest_does_not_carry(self) -> None:
        """Holding the attribution independently of the build arguments would list the origin of a
        model that is not included."""
        manifest = _extra_manifest()
        del manifest["models"][EXTRA_SECOND]
        card = _extra(manifest)

        assert UPSTREAM_MODELS[EXTRA_SECOND].source not in card
        assert UPSTREAM_MODELS[EXTRA_DEFAULT].source in card

    def test_its_usage_snippet_keeps_the_cfg_knobs_as_defaults(self, extra_card: str) -> None:
        """In a distribution whose default uses CFG, guidance / negative are shown as "optional
        defaults"."""
        assert "// guidanceScale: 4, // default" in extra_card
        assert "makes the negative prompt take effect" in extra_card

    def test_it_refuses_another_pipelines_manifest(self) -> None:
        manifest = _extra_manifest()
        manifest["models"][EXTRA_DEFAULT]["pipeline"] = "sbv2/1"

        with pytest.raises(ValueError):
            _extra(manifest)


class TestFrontmatter:
    def test_it_opens_with_a_yaml_block(self, card: str) -> None:
        lines = card.splitlines()
        assert lines[0] == "---"
        assert lines.index("---", 1) > 1

    def test_it_declares_the_fields_hf_reads(self, card: str) -> None:
        head = card.split("---")[1]
        assert "pipeline_tag: text-to-image" in head
        assert "base_model: circlestone-labs/Anima-Base-v1.0-Diffusers" in head
        assert "base_model_relation: quantized" in head
        assert "license: other" in head
        assert "license_name: circlestone-labs-non-commercial-license" in head
        assert (
            "license_link: https://huggingface.co/circlestone-labs/Anima/blob/"
            "f973fc41ec7545364ac9776c2440285f43ff2a30/LICENSE.md" in head
        )
        assert "library_name: karume" in head

    def test_the_license_link_is_pinned_to_a_commit(self, card: str) -> None:
        """The link shows the text bundled as `LICENSE.md` (v1.2 at
        `circlestone-labs/Anima@f973fc41`), not whatever `main` holds — a moving branch would
        drift from the bundled text the day upstream publishes another version."""
        head = card.split("---")[1]
        (link,) = [line for line in head.splitlines() if line.startswith("license_link: ")]
        assert "/blob/main/" not in link
        assert "/blob/f973fc41ec7545364ac9776c2440285f43ff2a30/" in link
        assert [line for line in head.splitlines() if line.startswith("  - ")] == [
            "  - text-to-image",
            "  - webgpu",
        ]

    def test_both_repositories_declare_the_same_upstream_license(
        self, card: str, extra_card: str
    ) -> None:
        """Both repositories are Derivatives of the same base — only one upstream license
        applies."""
        assert card.split("---")[1] == extra_card.split("---")[1]


class TestSections:
    def test_the_official_card_carries_every_section(self, card: str) -> None:
        headings = [line for line in card.splitlines() if line.startswith("## ")]
        assert headings == [
            "## What is this",
            "## Models and their origins",
            "## License",
            "## Models",
            "## Usage",
            f"## Model: {OFFICIAL_DEFAULT}",
            f"## Model: {OFFICIAL_SECOND}",
        ]

    def test_the_extra_card_carries_the_same_shape(self, extra_card: str) -> None:
        """The two cards have **the same skeleton** (only the title, the overview and the
        introduction of the origins section differ)."""
        headings = [line for line in extra_card.splitlines() if line.startswith("## ")]
        assert headings == [
            "## What is this",
            "## Models and their origins",
            "## License",
            "## Models",
            "## Usage",
            f"## Model: {EXTRA_DEFAULT}",
            f"## Model: {EXTRA_SECOND}",
        ]

    def test_each_model_section_carries_its_own_tables(self, card: str) -> None:
        _, _, rest = card.partition(f"## Model: {OFFICIAL_DEFAULT}")
        first, _, second = rest.partition(f"## Model: {OFFICIAL_SECOND}")
        for part in (first, second):
            assert [line for line in part.splitlines() if line.startswith("### ")] == [
                "### Quants",
                "### Defaults",
            ]

    def test_it_shows_the_minimal_typescript_entry_point(self, card: str) -> None:
        assert "await Pipeline.fromPretrained({" in card
        assert f'  repo: "{REPO}",' in card
        assert "@karume/models" in card
        assert "using pipeline" in card

    def test_the_usage_snippet_declares_the_pipeline_with_await_using(self, card: str) -> None:
        """`anima.Pipeline` has only `[Symbol.asyncDispose]` — a synchronous `using` fails at the
        declaration.

        A substring match (`"using pipeline" in card`) passes for both `await using` and
        `using`, so the whole declaration line is compared.
        """
        lines = card.splitlines()
        assert "await using pipeline = await Pipeline.fromPretrained({" in lines
        assert not any(line.startswith("using pipeline") for line in lines)

    def test_the_usage_snippet_offers_the_revision_pin(self, card: str) -> None:
        """The source is in object ref form — without a place to write the revision, readers
        implicitly follow `main`."""
        assert '  // revision: "<full commit sha>",' in card
        assert "Pin a commit for reproducible builds" in card

    def test_the_usage_snippet_carries_the_optional_knobs_commented_out(self, card: str) -> None:
        """The form of the 2026-08-12 ruling: the minimal working form + `generate()`'s optional
        arguments written alongside as comments (the values come from the manifest — both steps
        and resolution show this fake manifest's values as they are).
        """
        assert "  // steps: 7, //" in card
        assert "  // resolution: { width: 640, height: 384 }, // default" in card
        # guidance and negativePrompt mean something **as a pair** (with guidanceScale 1 the
        # latter is rejected).
        assert "  // guidanceScale: 5," in card
        assert '  // negativePrompt: "fake negative value",' in card


class TestModelSelection:
    """The axis that first became machine-readable in v2 (ADR 0041 §2) — it shows up in 3 places:
    the list, the default and the usage."""

    def test_it_lists_every_model_with_its_quants(self, card: str) -> None:
        _, _, rest = card.partition("## Models")
        table, _, _ = rest.partition("## Usage")
        rows = [line for line in table.splitlines() if line.startswith("| `")]
        assert rows == [
            f"| `{OFFICIAL_DEFAULT}` (default) | `anima/1` |"
            " `f16` / `f16+dit8-a8` / `f16-c16` | `f16+dit8-a8` |",
            f"| `{OFFICIAL_SECOND}` | `anima/1` | `f16+dit8` | `f16+dit8` |",
        ]

    def test_the_usage_snippet_names_the_default_model_and_its_quant(self, card: str) -> None:
        """The default is spelled out and **commented out** (the "just uncomment it" form of the
        2026-08-12 ruling), and the selectable values are listed from the manifest on the same
        line (they follow when models / quants are added).
        """
        available = " / ".join(sorted((OFFICIAL_DEFAULT, OFFICIAL_SECOND)))
        assert f'  // model: "{OFFICIAL_DEFAULT}", // default — available: {available}' in card
        assert (
            '  // quant: "f16+dit8-a8", // default — available: f16 / f16+dit8-a8 / f16-c16'
        ) in card

    def test_it_follows_the_manifest_when_the_default_moves(self) -> None:
        manifest = _official_manifest()
        manifest["defaultModel"] = OFFICIAL_SECOND
        card = _official(manifest)
        available = " / ".join(sorted((OFFICIAL_DEFAULT, OFFICIAL_SECOND)))

        assert f'  // model: "{OFFICIAL_SECOND}", // default — available: {available}' in card
        assert '  // quant: "f16+dit8", // default — available: f16+dit8' in card
        assert f"| `{OFFICIAL_SECOND}` (default) |" in card


class TestDerivation:
    """MUST: the numbers, download sizes and quant table come from the manifest (nothing
    hand-written mixed in)."""

    def test_each_model_section_counts_only_its_own_files(self, card: str) -> None:
        """Counted again per section — the default entry is 111 + i8 444 + rope part 333 + tokenizer
        555 = 1,443 B (= 1.41 KiB), the 2nd is 111 + i8 666 + tokenizer 555 = 1,332 B
        (= 1.30 KiB).
        """
        _, _, rest = card.partition(f"## Model: {OFFICIAL_DEFAULT}")
        first, _, second = rest.partition(f"## Model: {OFFICIAL_SECOND}")

        assert "| 1.41 KiB (666 B shared; 555 B of assets, read on the host) |" in first
        assert "| 1.30 KiB (666 B shared; 555 B of assets, read on the host) |" in second
        # The one-row-per-shard table was dropped (ruling of 2026-09-03) — no weight path
        # appears on the card.
        assert f"{OFFICIAL_DEFAULT}/transformer" not in card
        assert "shared/text_encoder/model.krm" not in card

    def test_it_counts_the_asset_part_in_every_seat_that_declares_it(self, card: str) -> None:
        """The dedicated part of the asset (the rope base table) is a downloaded file too — each of
        the two dtype entries has one."""
        manifest = _official_manifest()
        for entry in manifest["models"][OFFICIAL_DEFAULT]["weights"]["transformer"].values():
            del entry["container"]["parts"][-1]
        dropped = _official(manifest)

        assert "| 1.19 KiB (" in _quant_row(card, "f16")
        assert "| 888 B (" in _quant_row(dropped, "f16")
        assert "| 1.08 KiB (" in _quant_row(dropped, "f16+dit8-a8")

    def test_an_asset_is_counted_in_every_quant(self, card: str) -> None:
        """assets are unconditional files independent of the quant choice (ADR 0041 §3) — they count
        in every entry."""
        manifest = _official_manifest()
        manifest["models"][OFFICIAL_DEFAULT]["assets"] = {}
        dropped = _official(manifest)

        assert "555 B of assets, read on the host" in _quant_row(card, "f16")
        assert "| 666 B (" in _quant_row(dropped, "f16")
        assert "of assets" not in _quant_row(dropped, "f16+dit8-a8")

    def test_it_takes_the_sizes_from_the_manifest(self, card: str) -> None:
        manifest = _official_manifest()
        manifest["models"][OFFICIAL_DEFAULT]["weights"]["text_encoder"]["f16"]["container"][
            "parts"
        ][0]["size"] = 999
        moved = _official(manifest)
        assert "| 1.41 KiB (" in _quant_row(card, "f16+dit8-a8")
        assert "| 2.28 KiB (" in _quant_row(moved, "f16+dit8-a8")

    def test_it_marks_exactly_the_default_quant_of_each_model(self, card: str) -> None:
        """**One mark per model** (the owner of the quant table decides the default — ADR 0041
        §2)."""
        _, _, rest = card.partition(f"## Model: {OFFICIAL_DEFAULT}")
        first, _, second = rest.partition(f"## Model: {OFFICIAL_SECOND}")
        marked = {
            section.partition("### Quants")[2].partition("###")[0]: expected
            for section, expected in (
                (first, "| `f16+dit8-a8` (default) |"),
                (second, "| `f16+dit8` (default) |"),
            )
        }
        for quants, expected in marked.items():
            rows = [line for line in quants.splitlines() if line.startswith("| `")]
            default = [line for line in rows if "(default)" in line]
            assert len(default) == 1
            assert default[0].startswith(expected)
        # The table has no "default" column (the mark is only next to the name — no column of
        # blanks).
        assert "| Quant | What it is | Download | Weights | Compute |" in card

    def test_it_carries_every_quant_with_its_session_knobs(self, card: str) -> None:
        assert (
            "| `f16` | — | 1.19 KiB (666 B shared; 555 B of assets, read on the host) |"
            " `text_encoder` = `f16` / `transformer` = `f16` | — |" in card
        )
        assert "`linearCompute` = `a8`" in card
        assert "requires `shaderF16`" in card

    def test_it_prints_the_presentation_fields_the_manifest_carries(self, card: str) -> None:
        """The presentation fields appear on the card with the same strings (ADR 0075 decision 5 —
        the description does not grow in 2 places)."""
        assert "**fake label** — Fake description." in card

    def test_it_explains_the_abbreviation_the_seat_names_use(self, card: str) -> None:
        """The abbreviation mapping is **always** shown (ADR 0074 decision 4) — so that readers can
        tell which component `dit8` is about."""
        assert "In a quant name, `dit` is the `transformer` component." in card

    def test_it_takes_the_defaults_from_each_models_pipeline_config(self, card: str) -> None:
        assert "- **steps**: 7" in card
        assert "- **resolution**: 640 × 384" in card
        assert "fake negative value" in card
        assert "- **steps**: 4" in card
        assert "- **resolution**: 128 × 256" in card

    def test_it_only_warns_about_the_unused_negative_prompt_at_guidance_one(
        self, card: str
    ) -> None:
        """The note appears only for guidance 1 entries (in the fake manifest, the one of the
        default model)."""
        manifest = _official_manifest()
        manifest["models"][OFFICIAL_DEFAULT]["pipelineConfig"]["defaults"]["guidanceScale"] = 4

        assert "the negative prompt is not used" in card
        assert "the negative prompt is not used" not in _official(manifest)


class TestDeterminism:
    def test_it_renders_the_same_bytes_for_the_same_manifest(self) -> None:
        first = _official(_official_manifest())
        second = _official(json.loads(json.dumps(_official_manifest())))
        assert first.encode("utf-8") == second.encode("utf-8")

    def test_it_does_not_mutate_the_manifest(self) -> None:
        manifest = _official_manifest()
        before = copy.deepcopy(manifest)
        _official(manifest)
        assert manifest == before


class TestPipelineGate:
    def test_it_refuses_a_pipeline_it_does_not_describe(self) -> None:
        manifest = _official_manifest()
        manifest["models"][OFFICIAL_SECOND]["pipeline"] = "sbv2/1"
        with pytest.raises(ValueError, match="sbv2/1"):
            _official(manifest)


def _house_sentence(card: str) -> str:
    """The paragraph right under the card's title."""
    lines = card.splitlines()
    return lines[next(index for index, line in enumerate(lines) if line.startswith("# ")) + 2]


class TestTheHouseSentence:
    """The house non-endorsement sentence (`karume.modelcard.introduction`) beside the licence's
    own §3(d) sentence, which stays word for word in the License section."""

    def test_the_official_card_names_its_models_and_circlestone(self, card: str) -> None:
        assert _house_sentence(card) == (
            "This is an unofficial conversion of Anima Turbo v1.1 and Anima Aesthetic v1.1 for"
            " Karume. It is not affiliated with or endorsed by CircleStone Labs LLC."
        )

    def test_the_extra_card_names_each_fine_tune_and_its_author(self, extra_card: str) -> None:
        assert _house_sentence(extra_card) == (
            "This is an unofficial conversion of WAI-ANIMA v1.0 (base 1.0) and copycat-anima"
            " 20260610 for Karume. It is not affiliated with or endorsed by WAI0731 and calculater."
        )

    def test_the_licence_sentence_stays_as_it_is(self, card: str, extra_card: str) -> None:
        for text in (card, extra_card):
            assert (
                "- This is not an official product of CircleStone Labs LLC, and it is not endorsed,"
                " approved or\n  validated by CircleStone Labs LLC."
            ) in text
