"""The umT5-XXL encoder distribution recipe (`wan.umt5_distribution`) — assembling the distribution
repository `karume-umt5-xxl`.

The inputs are a synthetic container with the real spelling and the real per-kind storage
(`umt5_fixture`) and a synthetic tokenizer asset with the writer's top-level fields. The cases that
check the gate refuses something are built from the same fixture, **moving only the declarations
away from the real thing**.
Where the `karume-umt5-xxl` mirror is present, the real binding table is pinned down to its counts
(169 i8, 73 f32) and the plan is built from the real container and tokenizer asset as the mirror
ships them (SKIP without it).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import ir_fixtures
import pytest
from container_series import part_paths, write_component
from real_assets import mirror_asset, mirror_container
from upstream_fixture import OTHER_REVISION, stamp_fixture_provenance

import dist
from _shared.licenses import APACHE_LICENSE_2_0_PATH
from karume.container import Provenance
from karume.dist import (
    MANIFEST_FILENAME,
    MODEL_CARD_FILENAME,
    NOTICE_FILENAME,
    DistError,
    assemble_family,
    resolve_card_renderer,
)
from karume.modelcard import REVISION_PIN_COMMENT
from wan.sources import UMT5_SOURCES
from wan.tests import umt5_fixture
from wan.umt5_distribution import (
    PIPELINE,
    UMT5_CONSUMER_REPO_NAME,
    UMT5_DEFAULT_MODEL,
    UMT5_INPUTS,
    UMT5_KIND_LAYOUTS,
    UMT5_MODEL_FILE,
    UMT5_OUTPUT_PATHS,
    UMT5_PIPELINE,
    UMT5_RELATIVE_BIAS,
    UMT5_REPO_NAME,
    UMT5_ROLE,
    UMT5_SERIES,
    UMT5_SUPPORTED_PIPELINE,
    UMT5_TOKENIZER_FIELDS,
    UMT5_TOKENIZER_ROLE,
    UMT5_TOKENS,
    UMT5_WEIGHTS,
    render_umt5_model_card,
    storage_kind,
    umt5_bindings,
    umt5_container,
    umt5_plan,
    umt5_tokenizer_asset,
)

#: The row of the umT5 upstream table (the pin of the original `google/umt5-xxl` — ADR 0122
#: decision 1).
_UPSTREAM = UMT5_SOURCES[UMT5_DEFAULT_MODEL].source

#: The normal form of the provenance the writer (`wan.umt5_export.provenance`) bakes — the pinned
#: revision of the umT5 upstream.
_PINNED = Provenance(
    license=_UPSTREAM.license,
    notice=NOTICE_FILENAME,
    upstream_revision=_UPSTREAM.revision,
)


@pytest.fixture(autouse=True)
def _pinned_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the synthetic containers (`ir_fixtures` included) claim the provenance "baked from the
    revision the script pinned"."""
    stamp_fixture_provenance(monkeypatch, _PINNED)


def _series(root: Path, **overrides: Any) -> Path:
    """Put one container in the series and return its representative path (`overrides` are the
    fault-injection knobs of the synthetic container — the provenance is matched to the value
    `ir_fixtures` bakes and is swapped only for the duration of the test)."""
    container = umt5_container(root / "outputs" / "series")
    parts = umt5_fixture.umt5_container(provenance=ir_fixtures.FIXTURE_PROVENANCE, **overrides)
    write_component(container, parts)
    # A golden that must not enter the distribution (the series really holds these next to it).
    (container.parent / "reference.band-l0008.safetensors").write_bytes(b"golden")
    return container


def _tokenizer(root: Path, **overrides: Any) -> Path:
    """Put a tokenizer asset in the series and return its path
    (`umt5_fixture.write_tokenizer_asset`)."""
    return umt5_fixture.write_tokenizer_asset(root / "outputs" / "series", **overrides)


def _plan(root: Path, container: Path, model: str = UMT5_DEFAULT_MODEL):
    """Build the plan from the series' container and a normal-form tokenizer asset."""
    return umt5_plan(container, _tokenizer(root), model)


def _assemble(root: Path, container: Path) -> tuple[Path, dict[str, Any]]:
    out_dir = root / "models" / UMT5_REPO_NAME
    manifest = assemble_family(
        [_plan(root, container)],
        out_dir,
        UMT5_DEFAULT_MODEL,
        render_card=lambda manifest, host_assets: PIPELINE.card_profiles["umt5"](
            manifest, repo=f"hdae/{UMT5_REPO_NAME}", host_assets=host_assets
        ),
        root_files=PIPELINE.root_files,
    )
    return out_dir, manifest


@pytest.fixture
def assembled(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    return _assemble(tmp_path, _series(tmp_path))


def _model(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return manifest["models"][UMT5_DEFAULT_MODEL]


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


class TestLayout:
    def test_it_places_one_container_the_tokenizer_and_the_legal_texts(self, assembled) -> None:
        """One container (component `text_encoder`), one tokenizer asset and the four top-level
        files only — the series' golden does not enter."""
        out_dir, _ = assembled
        expected = [
            f"{UMT5_DEFAULT_MODEL}/{rel}" for rel in part_paths(UMT5_OUTPUT_PATHS[UMT5_ROLE])
        ]
        tokenizer = f"{UMT5_DEFAULT_MODEL}/{UMT5_OUTPUT_PATHS[UMT5_TOKENIZER_ROLE]}"
        assert _present(out_dir) == sorted(
            [
                *expected,
                tokenizer,
                MANIFEST_FILENAME,
                MODEL_CARD_FILENAME,
                "LICENSE.md",
                NOTICE_FILENAME,
            ]
        )
        assert UMT5_OUTPUT_PATHS[UMT5_ROLE] == "text_encoder/model.i8.krm"
        assert tokenizer == "xxl/tokenizer/tokenizer.json"

    def test_the_manifest_declares_the_encoder_seat_and_the_tokenizer(self, assembled) -> None:
        """The pipeline is `umt5-encoder/1`, one preset (i8), one asset (the tokenizer) and an
        empty config."""
        out_dir, manifest = assembled
        model = _model(manifest)
        assert manifest["defaultModel"] == UMT5_DEFAULT_MODEL == "xxl"
        assert model["pipeline"] == UMT5_PIPELINE == "umt5-encoder/1"
        assert list(model["weights"]) == [UMT5_ROLE]
        assert list(model["weights"][UMT5_ROLE]) == ["i8"]
        assert list(model["assets"]) == [UMT5_TOKENIZER_ROLE] == ["tokenizer"]
        tokenizer = model["assets"][UMT5_TOKENIZER_ROLE]
        assert tokenizer["path"] == "xxl/tokenizer/tokenizer.json"
        assert tokenizer["size"] == (out_dir / tokenizer["path"]).stat().st_size
        assert list(model["quants"]) == ["i8"]
        assert model["quants"]["i8"]["weights"] == {UMT5_ROLE: "i8"}
        assert model["quants"]["i8"]["session"] == {}
        assert model["defaultQuant"] == "i8"
        assert model["pipelineConfig"] == {}

    def test_the_repository_ships_the_apache_license_and_the_change_notice(self, assembled) -> None:
        out_dir, _ = assembled
        assert (out_dir / "LICENSE.md").read_bytes() == APACHE_LICENSE_2_0_PATH.read_bytes()
        prose = " ".join((out_dir / NOTICE_FILENAME).read_text(encoding="utf-8").split())
        assert "Apache License, Version 2.0" in prose
        assert "**int8 weights**" in prose
        assert "keep the source float32 values" in prose
        assert 'GELU(approximate="tanh")' in prose
        # The upstream storage is F32 (not a copy of bf16 — measured during the investigation).
        assert "holds the encoder in float32" in prose
        assert "bfloat16" not in prose
        # The upstream is the original encoder, and it was checked bit-identical to Wan2.1's
        # text_encoder (ADR 0122 decision 1 — the fact changed from "identity not checked").
        # The decoder and lm_head are not included.
        assert "has not been checked" not in prose
        assert "the encoder of the umT5-XXL checkpoint `google/umt5-xxl`" in prose
        assert "that repository has no NOTICE file" in prose
        assert "bit-identical to the float32 `text_encoder` folder" in prose
        assert "the decoder and `lm_head` are not included" in prose
        # The tokenizer is shipped on its own (the change notice of Apache 2.0 §4(b) — the
        # "not distributed" sentence is gone).
        assert "and a converted form of that checkpoint's tokenizer" in prose
        assert (
            "**Tokenizer**: the tokenizer files `tokenizer.json` and `tokenizer_config.json`"
            in prose
        )
        assert "without any text normalization" in prose
        assert "neither is the tokenizer" not in prose
        assert "original tokenizer files are not distributed here" in prose


class TestTheBindingGate:
    """The per-kind storage of the binding table (`assert_umt5_bindings`). A mix-up that a mere
    "contains i8" check lets through."""

    def test_it_refuses_quantized_relative_position_tables(self, tmp_path: Path) -> None:
        container = _series(tmp_path, table_layout="i8")
        with pytest.raises(DistError, match=UMT5_RELATIVE_BIAS):
            _plan(tmp_path, container)

    def test_it_refuses_linear_weights_left_in_f32(self, tmp_path: Path) -> None:
        container = _series(tmp_path, linear_layout="f32")
        with pytest.raises(DistError, match="'linear'"):
            _plan(tmp_path, container)

    def test_it_refuses_an_encoder_without_relative_position_tables(self, tmp_path: Path) -> None:
        container = _series(tmp_path, tables=False)
        with pytest.raises(DistError, match="weight kinds"):
            _plan(tmp_path, container)

    def test_the_bindings_are_counted_by_kind(self, tmp_path: Path) -> None:
        """The pair (not tautological): the normal form holds only the required storage per kind."""
        bindings = umt5_bindings(_series(tmp_path))
        layers = umt5_fixture.LAYERS
        assert bindings == {
            "constant": {"f32": 1},
            "embed_tokens": {"i8": 1},
            "linear": {"i8": layers},
            "norm": {"f32": layers + 1},
            UMT5_RELATIVE_BIAS: {"f32": layers},
        }
        assert all(set(layouts) == {UMT5_KIND_LAYOUTS[kind]} for kind, layouts in bindings.items())


class TestTheTokenizerGate:
    """The tokenizer asset gate (`assert_umt5_tokenizer`) — the tables escape the container checks,
    and a mix-up stays silent on the user's machine as "different id sequences"."""

    def test_the_plan_requires_the_tokenizer_asset(self, tmp_path: Path) -> None:
        container = _series(tmp_path)
        missing = umt5_tokenizer_asset(tmp_path / "outputs" / "series")
        with pytest.raises(DistError):
            umt5_plan(container, missing)

    def test_it_refuses_the_wan_tokenizer_asset(self, tmp_path: Path) -> None:
        """Wan's asset (with the preprocessing tables, with the provenance of Wan's checkpoint)
        fails on its format."""
        container = _series(tmp_path)
        with pytest.raises(DistError, match="karume-umt5-tokenizer/1"):
            umt5_plan(container, _tokenizer(tmp_path, format="karume-wan-umt5-tokenizer/1"))

    def test_it_refuses_extra_fields_such_as_prompt_cleaning(self, tmp_path: Path) -> None:
        container = _series(tmp_path)
        with pytest.raises(DistError, match="the fields"):
            umt5_plan(container, _tokenizer(tmp_path, promptClean={}))
        with pytest.raises(DistError, match="the fields"):
            umt5_plan(container, _tokenizer(tmp_path, scores=None))

    def test_it_refuses_a_tokenizer_from_another_revision_or_other_bytes(
        self, tmp_path: Path
    ) -> None:
        container = _series(tmp_path)
        row = UMT5_SOURCES[UMT5_DEFAULT_MODEL]
        source = {
            "repo": row.source.repo,
            "revision": OTHER_REVISION,
            "files": dict(sorted(row.tokenizer.items())),
        }
        with pytest.raises(DistError, match="the tokenizer provenance"):
            umt5_plan(container, _tokenizer(tmp_path, source=source))
        other_bytes = {
            **source,
            "revision": row.source.revision,
            "files": {"tokenizer.json": "0" * 64},
        }
        with pytest.raises(DistError, match="the tokenizer provenance"):
            umt5_plan(container, _tokenizer(tmp_path, source=other_bytes))

    def test_it_refuses_another_transformers_version(self, tmp_path: Path) -> None:
        container = _series(tmp_path)
        versions = {"tokenizers": "0.22.2", "transformers": "4.57.0"}
        with pytest.raises(DistError, match="transformers"):
            umt5_plan(container, _tokenizer(tmp_path, versions=versions))

    def test_it_refuses_another_max_length(self, tmp_path: Path) -> None:
        container = _series(tmp_path)
        with pytest.raises(DistError, match="maxLength"):
            umt5_plan(container, _tokenizer(tmp_path, maxLength=256))

    def test_the_synthetic_asset_has_the_writers_fields(self) -> None:
        """The synthetic asset has exactly the set of fields the gate requires (the pair — the
        normal form passes the gate because the fields line up)."""
        assert {
            "format",
            "source",
            "versions",
            "maxLength",
            "unkId",
            "eosId",
            "addedTokens",
            "space",
            "vocabText",
            "scores",
        } == UMT5_TOKENIZER_FIELDS


class TestTheContract:
    """The graph's inputs and outputs (`umt5_context_width`) — the reader binds the inputs by
    name."""

    @pytest.mark.parametrize(
        ("index", "replacement", "verified"),
        [
            (0, ("ids", "i32", [1, "L"]), True),
            # A graph whose embedding takes f32 token ids violates the IR acceptance rules — it
            # does not pass the public gate, so it is built deliberately with the low-level writer
            # (to see that the gate fails it on "the input dtype").
            (0, ("input_ids", "f32", [1, "L"]), False),
            (1, ("relative_position_buckets", "i32", ["L", "M"]), True),
        ],
        ids=["renamed", "dtype", "second-symbol"],
    )
    def test_it_refuses_other_inputs(
        self,
        tmp_path: Path,
        index: int,
        replacement: tuple[str, str, list[Any]],
        verified: bool,
    ) -> None:
        inputs = list(UMT5_INPUTS)
        inputs[index] = replacement
        container = _series(tmp_path, inputs=inputs, verified=verified)
        with pytest.raises(DistError, match="the graph inputs"):
            _plan(tmp_path, container)

    def test_it_returns_the_output_width(self, tmp_path: Path) -> None:
        from wan.umt5_distribution import assert_umt5_encoder

        assert assert_umt5_encoder(_series(tmp_path, width=12), UMT5_DEFAULT_MODEL) == 12


class TestTheProvenance:
    def test_it_refuses_a_container_baked_from_another_revision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stamp_fixture_provenance(
            monkeypatch,
            Provenance(
                license=_PINNED.license, notice=NOTICE_FILENAME, upstream_revision=OTHER_REVISION
            ),
        )
        with pytest.raises(DistError, match="different revision"):
            _plan(tmp_path, _series(tmp_path))

    def test_it_refuses_a_container_that_names_another_license(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stamp_fixture_provenance(
            monkeypatch,
            Provenance(
                license="cc-by-nc-4.0",
                notice=NOTICE_FILENAME,
                upstream_revision=_PINNED.upstream_revision,
            ),
        )
        with pytest.raises(DistError, match=r"provenance\.license is 'cc-by-nc-4.0'"):
            _plan(tmp_path, _series(tmp_path))

    def test_it_refuses_a_model_missing_from_the_upstream_table(self, tmp_path: Path) -> None:
        with pytest.raises(DistError, match="is unknown"):
            _plan(tmp_path, _series(tmp_path), "base")

    def test_the_upstream_is_the_pinned_commit_of_google_umt5_xxl(self) -> None:
        """Pins the main commit of the original model with 40 digits and holds the sha256 (64
        digits) of the three shards the encoder needs."""
        row = UMT5_SOURCES[UMT5_DEFAULT_MODEL]
        assert row.source.repo == "google/umt5-xxl"
        assert row.source.revision == "66cb9e7e85526fe440a945569e42c72fb6cbc0ad"
        assert row.source.license == "apache-2.0"
        assert sorted(row.shards) == [f"pytorch_model-0000{n}-of-00006.bin" for n in (1, 2, 3)]
        assert all(
            len(digest) == 64 and set(digest) <= set("0123456789abcdef")
            for digest in row.shards.values()
        )


class TestTheDriver:
    def test_the_driver_carries_the_pipeline(self) -> None:
        assert dist.PIPELINES["umt5"] is PIPELINE

    def test_the_default_output_is_the_repository_name(self) -> None:
        assert dist.default_out_dir(PIPELINE, [UMT5_DEFAULT_MODEL]).name == UMT5_REPO_NAME

    def test_the_card_profile_needs_no_choice(self) -> None:
        assert list(PIPELINE.card_profiles) == ["umt5"]
        assert resolve_card_renderer(PIPELINE, None) is render_umt5_model_card


class TestTheModelCard:
    def test_it_refuses_a_pipeline_it_does_not_describe(self, assembled) -> None:
        _, manifest = assembled
        foreign = json.loads(json.dumps(manifest))
        _model(foreign)["pipeline"] = "wan/1"
        with pytest.raises(ValueError, match=UMT5_SUPPORTED_PIPELINE):
            render_umt5_model_card(foreign, "hdae/x")

    def test_the_title_is_followed_by_the_house_sentence(self, assembled) -> None:
        _, manifest = assembled
        lines = render_umt5_model_card(manifest, "hdae/x").splitlines()
        title = lines.index("# umT5-XXL text encoder (int8) — Karume")
        assert lines[title + 2] == (
            "This is an unofficial conversion of google/umt5-xxl for Karume."
            " It is not affiliated with or endorsed by Google."
        )

    def test_it_attributes_the_encoder_of_google_umt5_xxl(self, assembled) -> None:
        """The attribution is the encoder of the original model's commit (the decoder and lm_head
        are not included), and the Relation is the checked relationship with Wan's two checkpoints
        (ADR 0122 decision 1). The original encoder sits at the repository root, so no empty
        subfolder is passed into the attribution text."""
        out_dir, _ = assembled
        card = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")
        prose = " ".join(card.split())
        assert "base_model: google/umt5-xxl" in card
        assert "base_model_relation: quantized" in card
        assert f"license: {_UPSTREAM.license}" in card
        assert f"at commit `{_UPSTREAM.revision}`" in card
        assert "the encoder weights (`shared.weight` and `encoder.*`) of [google/umt5-xxl]" in prose
        assert "The decoder and `lm_head` are not included." in prose
        assert "folder of [" not in prose
        assert "the `` folder" not in prose
        assert "every tensor is bit-identical to the float32 `text_encoder` folder" in prose
        assert "round-to-nearest-even rounding (checked on 2026-10-04)" in prose
        assert "base_model: Wan-AI/" not in card
        assert "has not been checked" not in prose

    def test_the_usage_is_the_standalone_encoder(self, assembled) -> None:
        """Usage is the standalone `Encoder` (`@karume/models/umt5`) — the source is the same
        object ref form as the other families plus the revision pin."""
        out_dir, _ = assembled
        lines = (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8").splitlines()
        start = lines.index('import { Encoder } from "jsr:@karume/models/umt5";')
        assert lines[start : start + 12] == [
            'import { Encoder } from "jsr:@karume/models/umt5";',
            "",
            "const encoder = await Encoder.fromPretrained({",
            f'  repo: "hdae/{UMT5_REPO_NAME}",',
            *REVISION_PIN_COMMENT,
            "});",
            'const { inputIds, hiddenStates } = await encoder.encode("a cat");'
            " // hiddenStates: [L, 4096] f32, row-major",
            "await encoder.dispose();",
            "```",
        ]
        card = " ".join(lines)
        assert "There is no standalone pipeline class" not in card

    def test_the_usage_states_the_input_rules(self, assembled) -> None:
        """The input rules (no preprocessing, six rejections — measured by
        `wan.umt5_xxl_tokenizer`)."""
        out_dir, _ = assembled
        card = " ".join((out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8").split())
        assert "no prompt cleaning and no Unicode normalization" in card
        assert "- characters outside the vocabulary;" in card
        assert "special tokens written in the text" in card
        assert "`▁` (U+2581) right after a space;" in card
        assert "whitespace other than the plain space U+0020" in card
        assert "and a trailing space" in card
        assert "- more than 512 tokens" in card
        assert "- fewer than 2 tokens (an empty text)." in card

    def test_wan22_is_named_only_as_a_consumer(self, assembled) -> None:
        """Wan appears only as a referrer under "Used by" (Wan2.1 is not published, so it is not
        named as a repository)."""
        out_dir, _ = assembled
        card = " ".join((out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8").split())
        assert "### Used by" in card
        assert f"[`hdae/{UMT5_CONSUMER_REPO_NAME}`]" in card
        assert "`Ti2vPipeline.fromPretrained` (`@karume/models/wan`)" in card
        assert "its own tokenizer asset adds Wan's prompt cleaning" in card
        assert "karume-wan2.1" not in card
        assert re.search(r"(?<!\w)Pipeline\.fromPretrained", card) is None

    def test_the_card_describes_the_tokenizer_and_the_buckets(self, assembled) -> None:
        out_dir, _ = assembled
        card = " ".join((out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8").split())
        assert "One asset, `tokenizer` (format `karume-umt5-tokenizer/1`)" in card
        assert "It carries no prompt cleaning." in card
        assert "the bucket of the distance `j − i` (key position minus query position)" in card
        assert "32 buckets and a maximum distance of 128" in card
        assert "start at 8, 12, 16, 23, 32, 46, 64 and 91" in card
        row = UMT5_SOURCES[UMT5_DEFAULT_MODEL]
        for file, digest in row.tokenizer.items():
            assert f"`{file}` (SHA-256 `{digest[:16]}…`)" in card
        assert "the tokenizer converted into one JSON table" in card

    def test_the_usage_explains_swapping_in_a_compatible_encoder(self, assembled) -> None:
        """The replacement paragraph states the spelling of the option, the pin rule, that it has no
        effect with precomputed, and the acceptance conditions (ADR 0122 decision 7, item 2)."""
        out_dir, _ = assembled
        card = " ".join((out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8").split())
        assert "`components: { text_encoder: { source: { repo, revision } } }`" in card
        assert "a bare repository name follows `main`" in card
        assert '`textEncoder: "precomputed"` never loads umT5, so a swap has no effect' in card
        assert "graph description has the same SHA-256" in card

    def test_the_inputs_and_the_token_range_are_named(self, assembled) -> None:
        out_dir, _ = assembled
        card = " ".join((out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8").split())
        low, high = UMT5_TOKENS
        assert "`input_ids` `[1, L]`" in card
        assert "`relative_position_buckets` `[L, L]`" in card
        assert "the encoder's last hidden states `[1, L, 4096]` in float32 out" in card
        assert f"({low} to {high}, the end token `</s>` included)" in card

    def test_the_device_limits_come_from_the_manifest(self, assembled) -> None:
        _, manifest = assembled
        assert "### Device limits" not in render_umt5_model_card(manifest, "hdae/x")
        changed = json.loads(json.dumps(manifest))
        _model(changed)["quants"]["i8"]["requiredLimits"] = {
            "maxBufferSize": 1_050_148_864,
            "maxStorageBufferBindingSize": 1_050_148_864,
        }
        card = render_umt5_model_card(changed, "hdae/x")
        assert "### Device limits" in card
        assert "`maxBufferSize` ≥ 1,050,148,864 bytes" in card


class TestTheWritersSpellTheSameNames:
    """The distribution recipe does not import torch, so it keeps its own spelling — it equals the
    writer's (which imports torch)."""

    def test_the_series_and_the_part_name(self) -> None:
        from wan import umt5_export

        assert umt5_export.SERIES_NAME == UMT5_SERIES
        assert umt5_export.MODEL_FILE == UMT5_MODEL_FILE
        assert umt5_export.COMPONENT_DIR == UMT5_ROLE
        assert umt5_export.GRAPH_NAME == UMT5_ROLE

    def test_the_writer_bakes_the_row_the_gate_checks(self) -> None:
        """The provenance the writer bakes and the provenance the distribution gate compares are
        the same row of the same table."""
        from wan import umt5_export

        assert umt5_export.provenance(UMT5_DEFAULT_MODEL) == _PINNED

    def test_the_graph_inputs_and_the_table_attribute(self) -> None:
        from wan import umt5_patch

        assert tuple(name for name, _, _ in UMT5_INPUTS) == umt5_patch.INPUT_NAMES
        assert all(umt5_patch.SYMBOL in shape for _, _, shape in UMT5_INPUTS)
        assert umt5_patch.RELATIVE_BIAS_ATTRIBUTE == UMT5_RELATIVE_BIAS

    def test_the_writer_classifies_with_the_same_function(self) -> None:
        """There is one classification of kinds (the writer's acceptance table and the assembly
        gate call the same function)."""
        from wan import umt5_export

        assert umt5_export.storage_kind is storage_kind

    def test_the_token_range_is_the_tokenizer_range(self) -> None:
        from wan import umt5_tokenizer

        assert UMT5_TOKENS == (umt5_tokenizer.MIN_TOKENS, umt5_tokenizer.MAX_LENGTH)

    def test_the_consumer_is_the_published_wan_repository(self) -> None:
        """The referrer named for the cross-repository reference is the published Wan2.2
        distribution repository."""
        from wan.distribution import WAN22_REPO_NAME

        assert UMT5_CONSUMER_REPO_NAME == WAN22_REPO_NAME


class TestStorageKind:
    @pytest.mark.parametrize(
        ("key", "kind"),
        [
            ("encoder.embed_tokens.weight", "embed_tokens"),
            ("encoder.block.3.layer.0.SelfAttention.q.weight", "linear"),
            ("encoder.block.3.layer.1.DenseReluDense.wi_0.weight", "linear"),
            ("encoder.block.3.layer.0.layer_norm.weight", "norm"),
            ("encoder.final_layer_norm.weight", "norm"),
            (
                "encoder.block.3.layer.0.SelfAttention.relative_attention_bias.weight",
                UMT5_RELATIVE_BIAS,
            ),
            ("const.e7554f9eb93a2e4e", "constant"),
        ],
    )
    def test_it_names_the_kind(self, key: str, kind: str) -> None:
        assert storage_kind(key) == kind


def _shipped_container() -> Path:
    """The i8 encoder container the `karume-umt5-xxl` mirror ships (its only storage label)."""
    (label,) = UMT5_WEIGHTS[UMT5_ROLE]
    return mirror_container(UMT5_REPO_NAME, UMT5_DEFAULT_MODEL, UMT5_ROLE, label)


class TestTheRealSeries:
    """The real container and tokenizer asset (written by `wan.umt5_export` and
    `wan.umt5_xxl_tokenizer`, read as the `karume-umt5-xxl` mirror ships them — byte-identical,
    ADR 0139 decision 1) pass every gate of the plan, and the binding table has decision 5's shape.

    Only the container's two documents, its binding table and the asset JSON are read (no weight
    payload). SKIP without the mirror.
    """

    def test_the_plan_passes_every_gate(self) -> None:
        container = _shipped_container()
        tokenizer = mirror_asset(UMT5_REPO_NAME, UMT5_DEFAULT_MODEL, UMT5_TOKENIZER_ROLE)
        plan = umt5_plan(container, tokenizer)
        assert plan.pipeline == UMT5_PIPELINE
        assert set(plan.artifacts) == {UMT5_ROLE, UMT5_TOKENIZER_ROLE}

    def test_the_binding_table_is_169_int8_and_73_float32_weights(self) -> None:
        """linear 168 + the vocabulary embedding 1 are i8, the relative-position tables 24 +
        RMSNorm 49 are f32 (24 layers — ADR 0119 addendum 10b). The lifted constants (`const.*`)
        are not weights, so they are not counted."""
        bindings = umt5_bindings(_shipped_container())
        assert bindings == {
            "constant": {"f32": 2},
            "embed_tokens": {"i8": 1},
            "linear": {"i8": 168},
            "norm": {"f32": 49},
            UMT5_RELATIVE_BIAS: {"f32": 24},
        }
        weights = {kind: layouts for kind, layouts in bindings.items() if kind != "constant"}
        assert sum(layouts.get("i8", 0) for layouts in weights.values()) == 169
        assert sum(layouts.get("f32", 0) for layouts in weights.values()) == 73
