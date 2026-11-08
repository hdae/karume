"""Unit tests of distribution directory assembly (`karume.dist`).

No real assets are used. The inputs plugged into the weights fields must, however, be **valid IR
containers** — assembly rejects, at the plan's entry point, a plan whose weights role points at a
generated product (`payload`) (`dist.ModelPlan.__post_init__`), and checks the input containers
against every rule of IR v1 before placing them (`dist.assert_weight_components_verified`). They
are synthesized by {@link ir_fixtures} (one graph of a few KB), and only the `assets` field can
hold raw bytes (the `extras` field was retired in `karume/5`).

Since manifest v2 (`karume/2` — ADR 0041) the repository layout is uniformly "per-model subtree +
`shared/`", so every expected path starts with `<model name>/…`.

NOTE: every per-family distribution recipe has moved out of the wheel (ADR 0065 stages 3+4 done),
so the cases that depend on them live in `tools/export-recipes/<family>/tests/test_distribution.py`.
What remains here is only the layer observable with the core alone — the size limits a synthetic
plan suffices for, the complete quant mapping, the staging/swap invariants, the resolution rules of
the attributed profile, and the convention of taking the accepted set from the caller.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import pytest
from conftest import REPO_ROOT
from container_fixture import write_f16_steps_container
from ir_fixtures import ir_container as _ir_container
from ir_fixtures import ir_parts, with_an_unknown_op

from karume import dist, limits, verify
from karume.artifacts import SUPERSEDED_SUFFIX
from karume.container import (
    BLOCK_MAX_BYTES,
    HEADER_BYTES,
    MAX_DESCRIPTOR_BYTES,
    MAX_PARTS,
    PART_MAX_BYTES,
    canonical_json,
    numbered_name,
    read_descriptor_refs,
    read_header,
    write_header,
)
from karume.dist import (
    MANIFEST_FILENAME,
    MANIFEST_FORMAT,
    MAX_QUANT_DESCRIPTION_CHARS,
    MAX_QUANT_LABEL_CHARS,
    MODEL_CARD_FILENAME,
    PIPELINES,
    SHARED_DIRNAME,
    STORAGE_DTYPE_LABELS,
    Artifact,
    CrossRepoReference,
    DistError,
    ExternalComponents,
    ModelPlan,
    Pipeline,
    WeightFiles,
    assemble_family,
    assert_manifest_limits,
    assert_model_name,
    assert_quant_presentation,
    build_parser,
    complete_quant_weights,
    declared_references,
    is_external_ref,
    main,
    manifest_text,
    materialize,
    parse_reference_revisions,
    preprocessor_channels,
    resolve_card_renderer,
    resolve_repo,
    verify_dist,
)
from karume.modelcard import FileRefKey


def _write_series(root: Path, payloads: Sequence[bytes], name: str = "model.krm") -> Path:
    """Writes a part sequence into a series directory with numbered names and returns **the
    representative path**.

    A distribution is always split (ADR 0081), so the series-side files are always numbered too —
    the representative path itself is not written (writing it would make "a single file
    coexisting with numbered ones", and assembly fails loudly).
    """
    root.mkdir(parents=True, exist_ok=True)
    total = len(payloads)
    for index, payload in enumerate(payloads, start=1):
        (root / numbered_name(name, index, total)).write_bytes(payload)
    return root / name


#: The weights key (= component name) of the synthetic plans. Every field `_synthetic_plan` sets
#: uses this one word.
COMPONENT = "w"


def ir_container(*, mark: str, named: str = COMPONENT, **extra: Any) -> list[bytes]:
    """Wraps {@link ir_fixtures.ir_container} with a default that **names it after the
    component**.

    Assembly checks "the container's graph name == the weights key"
    ({@link karume.dist.assert_weight_components_verified}), so keeping the spelling derived from
    `mark` (`weights-D1` and so on) would give an input of a form the real thing never has.
    `mark` stays the tensor key prefix (= what makes the bytes differ), and only the graph name is
    matched to the field's key.
    """
    return _ir_container(mark=mark, named=named, **extra)


def _ref(path: str) -> dict[str, Any]:
    """Fake values of the three fields (the size-limit checks only count entries, so the form alone
    suffices)."""
    return {"path": path, "size": 1, "sha256": "0" * 64}


def _present(out_dir: Path) -> list[str]:
    return sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())


def _synthetic_series(root: Path, mark: str, named: str = COMPONENT) -> Path:
    """Writes a valid IR container series that differs per `mark` into `root/<mark>/` and returns
    **the representative path**.

    A weights field can point only at a placement (`source`) (`dist.ModelPlan.__post_init__`), so
    synthetic plans also write the series before pointing at it. Which model claims which bytes is
    told apart by the key prefix of the weight tensors (`mark`) — **`mark`s of the same length give
    sequences of the same size that differ only in content**, and a different length changes the
    size too (a tool for observing the pre-filter of the sharing check).
    """
    return _write_series(root / mark, ir_container(mark=mark, named=named))


def _part_rel_paths(rel_path: str, mark: str, named: str = COMPONENT) -> list[str]:
    """The relative paths **within the model subtree** where the series of {@link _synthetic_series}
    lands (in declaration order).

    A distribution is always split (container-v1 §8), so one role is always a part sequence of
    several elements.
    """
    total = len(ir_container(mark=mark, named=named))
    return [numbered_name(rel_path, index, total) for index in range(1, total + 1)]


def _part_refs(
    prefix: str, rel_path: str, mark: str, named: str = COMPONENT
) -> list[dict[str, Any]]:
    """The three fields of {@link _synthetic_series} placed under `prefix/` (in declaration
    order)."""
    return [
        {
            "path": f"{prefix}/{rel}",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for rel, payload in zip(
            _part_rel_paths(rel_path, mark, named),
            ir_container(mark=mark, named=named),
            strict=True,
        )
    ]


def _part_paths(prefix: str, rel_path: str, mark: str, named: str = COMPONENT) -> list[str]:
    """Only the paths of {@link _part_refs} (in the order compared with the return value of
    `verify_dist`)."""
    return [ref["path"] for ref in _part_refs(prefix, rel_path, mark, named)]


def _part_sizes(prefix: str, rel_path: str, mark: str, named: str = COMPONENT) -> dict[str, int]:
    """`{path: size}` of {@link _part_refs} (exactly the form `verify_dist` returns)."""
    return {ref["path"]: ref["size"] for ref in _part_refs(prefix, rel_path, mark, named)}


def _synthetic_plan(root: Path, name: str, rel_path: str, mark: str) -> ModelPlan:
    """The smallest plan with one role only (a synthetic plan for observing only placement and the
    folding of shared files).

    The weights field points at the real IR container series {@link _synthetic_series} wrote.
    """
    return ModelPlan(
        name=name,
        pipeline="anima/1",
        artifacts={"w": Artifact(rel_path=rel_path, source=_synthetic_series(root, mark))},
        weights={"w": {"f16": WeightFiles(file="w")}},
        assets={},
        quants={"f16": {"weights": {"w": "f16"}, "session": {}}},
        default_quant="f16",
        pipeline_config={},
    )


class TestModelName:
    """A model name is both a manifest key and a directory name in the repository (ADR 0041 §6 /
    §9)."""

    @pytest.mark.parametrize("name", ["../escape", "with space", ".hidden", "a/b", ""])
    def test_it_refuses_a_name_that_is_not_a_path_segment(self, name: str) -> None:
        with pytest.raises(DistError, match="allowed characters"):
            assert_model_name(name)

    def test_it_refuses_the_name_reserved_for_shared_files(self) -> None:
        with pytest.raises(DistError, match=SHARED_DIRNAME):
            assert_model_name(SHARED_DIRNAME)

    def test_it_accepts_the_names_the_distributions_actually_use(self) -> None:
        """The spelling forms the existing recipes use (lowercase letters / digits / hyphen / dot)
        pass.

        NOTE: the constants are not imported from the recipes — the core's tests do not flow back
        into the families (ADR 0065 decision 3). Each recipe's `tests/test_distribution.py`
        checks with its own constants that the spellings fit the real thing.
        """
        for name in ("small", "base", "so400m", "v4-small", "crnn-epoch3", "jvnv-F1"):
            assert assert_model_name(name) == name


class TestQuantCompletion:
    """The hub checks a quant's `weights` as **a complete mapping** (ADR 0041 §3)."""

    _weights: ClassVar[dict] = {
        "text_encoder": {"i8": WeightFiles("text_encoder")},
        "front": {"f16": WeightFiles("front_f16"), "i8": WeightFiles("front_i8")},
    }

    def test_it_fills_in_the_weights_that_have_a_single_dtype(self) -> None:
        completed = complete_quant_weights(
            self._weights, {"w8": {"weights": {"front": "i8"}, "session": {}}}
        )
        assert completed["w8"]["weights"] == {"front": "i8", "text_encoder": "i8"}

    def test_it_orders_the_mapping_by_the_weights_declaration(self) -> None:
        """If filled-in entries piled up at the end, the same roles would appear in different orders
        in the weights section and the quant section."""
        completed = complete_quant_weights(
            self._weights, {"w8": {"weights": {"front": "i8"}, "session": {}}}
        )
        assert list(completed["w8"]["weights"]) == ["text_encoder", "front"]

    def test_it_keeps_the_rest_of_the_quant_untouched(self) -> None:
        quants = {
            "w8a8": {
                "weights": {"front": "i8"},
                "session": {"linearCompute": "a8"},
                "gpuFeatures": {"shaderF16": True},
            }
        }
        completed = complete_quant_weights(self._weights, quants)
        assert completed["w8a8"]["session"] == {"linearCompute": "a8"}
        assert completed["w8a8"]["gpuFeatures"] == {"shaderF16": True}

    def test_it_refuses_to_pick_a_dtype_when_the_weights_offer_a_choice(self) -> None:
        """Only entries with a single choice may be filled in — picking a default on our own would
        silently ship a different storage form."""
        with pytest.raises(DistError, match=r"has several dtypes .* but none is specified"):
            complete_quant_weights(self._weights, {"w8": {"weights": {}, "session": {}}})

    def test_it_refuses_an_unknown_weights_name(self) -> None:
        with pytest.raises(DistError, match="unknown weights 'voice'"):
            complete_quant_weights(
                self._weights, {"w8": {"weights": {"voice": "i8"}, "session": {}}}
            )

    def test_it_refuses_a_dtype_the_weights_do_not_carry(self) -> None:
        with pytest.raises(DistError, match="has no dtype 'bf16'"):
            complete_quant_weights(
                self._weights, {"w8": {"weights": {"front": "bf16"}, "session": {}}}
            )


class TestManifestLimits:
    """The hub rejects with the same size limits (ADR 0041 §7) — nothing ships only to fail
    afterwards."""

    @staticmethod
    def _container() -> dict[str, Any]:
        """The smallest container entry that passes the size-limit checks (the length of part 0
        follows from the two documents)."""
        graph, model = 40, 24
        return {
            "descriptor": {
                "graph": {"length": graph, "sha256": "0" * 64},
                "model": {"length": model, "sha256": "1" * 64},
            },
            "parts": [
                {"path": "m/w-00001-of-00002.krm", "size": 24 + graph + model, "sha256": "2" * 64},
                {"path": "m/w-00002-of-00002.krm", "size": 64, "sha256": "3" * 64},
            ],
        }

    @staticmethod
    def _model(config: Any = None) -> dict[str, Any]:
        return {
            "pipeline": "anima/1",
            "weights": {"w": {"f16": {"container": TestManifestLimits._container()}}},
            "assets": {},
            "quants": {"f16": {"weights": {"w": "f16"}, "session": {}}},
            "defaultQuant": "f16",
            "pipelineConfig": {} if config is None else config,
        }

    def _manifest(self, models: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "format": MANIFEST_FORMAT,
            "generator": "karume/9.9.9",
            "defaultModel": next(iter(models)),
            "models": dict(models),
        }

    def test_it_accepts_the_largest_repository_the_hub_reads(self) -> None:
        models = {f"m{index}": self._model() for index in range(32)}
        assert_manifest_limits(self._manifest(models))

    def test_it_refuses_one_model_beyond_the_limit(self) -> None:
        models = {f"m{index}": self._model() for index in range(33)}
        with pytest.raises(DistError, match="models has 33 entries"):
            assert_manifest_limits(self._manifest(models))

    def test_it_refuses_a_pipeline_config_beyond_the_limit(self) -> None:
        oversized = {"table": {f"k{index}": index for index in range(40_000)}}
        with pytest.raises(DistError, match="pipelineConfig"):
            assert_manifest_limits(self._manifest({"m": self._model(oversized)}))


class TestPartDeclaration:
    """The weights of `karume/5` have **a container entry** per dtype (the field of ADR 0109
    decision 3).

    This checks **the form of the field** (no v2 `file` left over, the three fields reach the
    files, the value range of the list); resolving from the files how many parts there are is
    {@link TestPartExpansion}.
    """

    _MARK = "weights-A"
    _REL_PATH = "w/model.krm"

    def _assemble(self, tmp_path: Path) -> tuple[Path, dict[str, Any]]:
        out_dir = tmp_path / "models" / "partitioned"
        plans = [_synthetic_plan(tmp_path / "series", "A", self._REL_PATH, self._MARK)]
        return out_dir, assemble_family(plans, out_dir, "A")

    def _rewrite_weights(self, out_dir: Path, entry: Any) -> None:
        """Swaps only the manifest of the placed distribution for another form (the files stay as
        they are)."""
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        manifest["models"]["A"]["weights"]["w"]["f16"] = entry
        (out_dir / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def test_the_format_identifier_names_the_fifth_manifest_version(self) -> None:
        """The format identifier cannot differ from the hub's by a single character (the hub reads
        one form only).

        `karume/5` is the form that replaced the dtype entries of weights with **container
        entries** (ADR 0109 decision 3) — old majors are not read, so forgetting the bump makes
        the hub fail with unsupported format.
        """
        assert MANIFEST_FORMAT == "karume/5"

    def test_it_declares_the_container_as_a_part_list_of_three_point_sets(
        self, tmp_path: Path
    ) -> None:
        out_dir, manifest = self._assemble(tmp_path)
        entry = manifest["models"]["A"]["weights"]["w"]["f16"]

        # No v2 `file` is left (with two forms side by side the hub would have to decide "which
        # one to read").
        assert list(entry) == ["container"]
        assert entry["container"]["parts"] == _part_refs("A", self._REL_PATH, self._MARK)
        for ref, payload in zip(
            entry["container"]["parts"], ir_container(mark=self._MARK), strict=True
        ):
            assert (out_dir / ref["path"]).read_bytes() == payload

    def test_the_parts_are_covered_by_the_declaration_check(self, tmp_path: Path) -> None:
        """The comparison follows the part list to the files (if moving to a list started to slip
        through, they would count as undeclared)."""
        out_dir, _ = self._assemble(tmp_path)
        assert verify_dist(out_dir) == _part_sizes("A", self._REL_PATH, self._MARK)

    def test_it_refuses_a_manifest_that_kept_the_v2_single_file_form(self, tmp_path: Path) -> None:
        """A manifest whose format identifier alone is v3 while its content is `{file}` cannot be
        read by the hub, so it fails here."""
        out_dir, manifest = self._assemble(tmp_path)
        ref = manifest["models"]["A"]["weights"]["w"]["f16"]["container"]["parts"][0]
        self._rewrite_weights(out_dir, {"file": ref})
        with pytest.raises(DistError, match=r"fields are not \['container'\]"):
            verify_dist(out_dir)

    @pytest.mark.parametrize("parts", [[], "A/w/model.krm"])
    def test_it_refuses_a_part_list_that_is_not_a_two_element_array(
        self, tmp_path: Path, parts: Any
    ) -> None:
        """Neither an empty list (= a dtype entry pointing at no part) nor a value that is not even
        a list is accepted.

        `container.parts` MUST have **two or more elements**, part 0 (descriptor) + part 1
        (const) (ADR 0109 decision 3 / container-v1 §8).
        """
        out_dir, manifest = self._assemble(tmp_path)
        entry = manifest["models"]["A"]["weights"]["w"]["f16"]["container"]
        self._rewrite_weights(out_dir, {"container": {**entry, "parts": parts}})
        with pytest.raises(DistError, match="is not an array of two or more entries"):
            verify_dist(out_dir)


class TestTheContainerEntryLimits:
    """The limits and rules on one `container` entry of `karume/5` (ADR 0109 decision 6).

    This function is **the only Python-side mirror** of the hub's parser (`parseContainer` in
    `packages/hub/src/manifest.ts`); a loosening would first surface on the user's side as "a
    manifest that was written but the hub cannot read". Fault injection on each rule pins that
    every one of them actually fails.

    MUST: a missing field also fails with `DistError` (not a bare indexing `KeyError` /
    `TypeError`) — {@link karume.dist.verify_dist} is the check that takes **a `karume.json` from
    outside**, so an exception outside the assembly's vocabulary would make "a broken manifest"
    indistinguishable from "a bug in the check".
    """

    _MARK = "limits-A"
    _REL_PATH = "w/model.krm"

    def _assemble(self, tmp_path: Path) -> tuple[Path, dict[str, Any]]:
        out_dir = tmp_path / "models" / "limits"
        plans = [_synthetic_plan(tmp_path / "series", "A", self._REL_PATH, self._MARK)]
        return out_dir, assemble_family(plans, out_dir, "A")

    @staticmethod
    def _rewrite(out_dir: Path, mutate) -> None:
        """Rewrites only the container entry of the placed `karume.json` (the files stay as they
        are)."""
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        mutate(manifest["models"]["A"]["weights"]["w"]["f16"]["container"])
        (out_dir / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _zero_length(container: dict[str, Any]) -> None:
        container["descriptor"]["graph"]["length"] = 0

    @staticmethod
    def _oversized_descriptor(container: dict[str, Any]) -> None:
        container["descriptor"]["model"]["length"] = MAX_DESCRIPTOR_BYTES + 1

    @staticmethod
    def _too_many_parts(container: dict[str, Any]) -> None:
        container["parts"] = [container["parts"][0]] * (MAX_PARTS + 1)

    @staticmethod
    def _part0_size_off_by_one(container: dict[str, Any]) -> None:
        container["parts"][0]["size"] += 1

    @staticmethod
    def _empty_part_with_a_wrong_digest(container: dict[str, Any]) -> None:
        container["parts"][1] = {**container["parts"][1], "size": 0, "sha256": "f" * 64}

    @staticmethod
    def _oversized_part(container: dict[str, Any]) -> None:
        container["parts"][-1] = {**container["parts"][-1], "size": PART_MAX_BYTES + 1}

    @staticmethod
    def _mixed_cross_repo(container: dict[str, Any]) -> None:
        container["parts"][-1] = {
            **container["parts"][-1],
            "repo": "hdae/other",
            "revision": "a" * 40,
        }

    @staticmethod
    def _missing_descriptor_length(container: dict[str, Any]) -> None:
        del container["descriptor"]["graph"]["length"]

    @staticmethod
    def _missing_part_size(container: dict[str, Any]) -> None:
        del container["parts"][1]["size"]

    @staticmethod
    def _part_ref_is_not_an_object(container: dict[str, Any]) -> None:
        container["parts"][1] = "A/w/model-00002-of-00003.krm"

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            ("_zero_length", "length is not a positive integer"),
            ("_oversized_descriptor", "is over the limit"),
            ("_too_many_parts", "limit"),
            ("_part0_size_off_by_one", r"size \d+ of part 0"),
            ("_empty_part_with_a_wrong_digest", "the sha256 of size 0"),
            ("_oversized_part", "the part length ceiling"),
            ("_mixed_cross_repo", "cross-repository references are mixed within the container"),
            ("_missing_descriptor_length", "length is not a positive integer"),
            ("_missing_part_size", "size is not a non-negative integer"),
            ("_part_ref_is_not_an_object", "is not an object"),
        ],
    )
    def test_a_rule_break_fails_loudly(self, tmp_path: Path, mutate: str, message: str) -> None:
        out_dir, _ = self._assemble(tmp_path)
        self._rewrite(out_dir, getattr(self, mutate))

        with pytest.raises(DistError, match=message):
            verify_dist(out_dir)

    def test_the_assembled_manifest_passes_every_rule(self, tmp_path: Path) -> None:
        """Counterpart — the manifest assembly wrote passes as is (the check is not "always
        fails")."""
        out_dir, _ = self._assemble(tmp_path)

        assert verify_dist(out_dir)


class TestTheManifestFieldsVerifyDistReads:
    """`verify_dist` also fails with `DistError` on missing or mistyped model / quant / asset
    fields.

    What lies outside the container entry (`weights` / `quants` / `defaultQuant` / the three
    fields of an asset) is also part of a `karume.json` from outside, so no bare indexing
    `KeyError` / `TypeError` leaks (the same reason as the MUST of
    {@link karume.dist.assert_container_limits}).
    """

    def _assemble(self, tmp_path: Path) -> Path:
        out_dir = tmp_path / "models" / "fields"
        plans = [_synthetic_plan(tmp_path / "series", "A", "w/model.krm", "fields-A")]
        assemble_family(plans, out_dir, "A")
        return out_dir

    @staticmethod
    def _rewrite(out_dir: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
        """Rewrites only the section of model A in the placed `karume.json` (the files stay as they
        are)."""
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        mutate(manifest["models"]["A"])
        (out_dir / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _missing_quants(model: dict[str, Any]) -> None:
        del model["quants"]

    @staticmethod
    def _missing_weights(model: dict[str, Any]) -> None:
        del model["weights"]

    @staticmethod
    def _dtype_entry_is_a_string(model: dict[str, Any]) -> None:
        model["weights"]["w"]["f16"] = "A/w/model.krm"

    @staticmethod
    def _missing_default_quant(model: dict[str, Any]) -> None:
        del model["defaultQuant"]

    @staticmethod
    def _missing_quant_weights(model: dict[str, Any]) -> None:
        del model["quants"]["f16"]["weights"]

    @staticmethod
    def _asset_without_a_path(model: dict[str, Any]) -> None:
        model["assets"]["tokenizer"] = {"size": 1, "sha256": "0" * 64}

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            ("_missing_quants", r"A\.quants is not an object"),
            ("_missing_weights", r"A\.weights is not an object"),
            ("_dtype_entry_is_a_string", r"A\.weights\.w\.f16 is not an object"),
            ("_missing_default_quant", r"A\.defaultQuant 'None' is not in quants"),
            ("_missing_quant_weights", r"A\.quants\.f16\.weights is not an object"),
            (
                "_asset_without_a_path",
                r"models\.A\.assets\.tokenizer\.path is not a non-empty string",
            ),
        ],
    )
    def test_a_missing_or_mistyped_field_fails_as_a_dist_error(
        self, tmp_path: Path, mutate: str, message: str
    ) -> None:
        out_dir = self._assemble(tmp_path)
        self._rewrite(out_dir, getattr(self, mutate))

        with pytest.raises(DistError, match=message):
            verify_dist(out_dir)


class TestDtypeLabelVocabulary:
    """The dtype labels of weights are inside the runtime's **storage dtype vocabulary**
    ({@link karume.dist.STORAGE_DTYPE_LABELS}).

    Labels are free keys of the manifest and the hub sees them only as strings, but the model
    card's body states that the labels are "the storage dtype vocabulary" (`modelcard.quants`; the
    card's note that `I4` cannot be opened with the official safetensors library, which used to be
    decided by **the label's spelling**, was retired with ADR 0109 decision 3). Without restricting
    the vocabulary, a family that spells a distribution containing i4 with the label `w4` would
    leave that claim of the card without backing.
    """

    _MARK = "weights-A"
    _REL_PATH = "w/model.krm"

    def _distribution(self, tmp_path: Path, label: str) -> Path:
        """Rewrites only the manifest of the placed distribution to the dtype entry `label` and
        returns it."""
        out_dir = tmp_path / "models" / "labelled"
        plan = _synthetic_plan(tmp_path / "series", "A", self._REL_PATH, self._MARK)
        assemble_family([plan], out_dir, "A")
        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        model = manifest["models"]["A"]
        model["weights"]["w"] = {label: model["weights"]["w"]["f16"]}
        model["quants"]["f16"]["weights"]["w"] = label
        (out_dir / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return out_dir

    def test_it_refuses_a_label_outside_the_storage_dtype_vocabulary(self, tmp_path: Path) -> None:
        """A spelling that carries a preset name into a dtype label (`w4`) fails loudly — the
        message lists the allowed vocabulary."""
        out_dir = self._distribution(tmp_path, "w4")

        with pytest.raises(DistError, match="dtype label 'w4'") as error:
            verify_dist(out_dir)
        assert "'i4'" in str(error.value)

    def test_it_accepts_the_packed_int4_label(self, tmp_path: Path) -> None:
        """`i4` is inside the vocabulary (the spelling the card's dtype-vocabulary line names)."""
        out_dir = self._distribution(tmp_path, "i4")

        assert verify_dist(out_dir) == _part_sizes("A", self._REL_PATH, self._MARK)

    def test_it_accepts_the_packed_int2_label(self, tmp_path: Path) -> None:
        """`i2` (ADR 0097's INT2 storage) is inside the vocabulary too — a distribution that
        declares INT2 can be assembled."""
        out_dir = self._distribution(tmp_path, "i2")

        assert verify_dist(out_dir) == _part_sizes("A", self._REL_PATH, self._MARK)

    def test_the_label_vocabulary_covers_every_storage_dtype_of_the_ir_gate(self) -> None:
        """The label vocabulary contains the storage dtype vocabulary of the IR verifier
        (mechanizing a copy that fails to keep up).

        `STORAGE_DTYPE_LABELS` is a hand-made copy of the decode-path vocabulary of the codec
        ledger (`CodecLayout`), and the same package has a second copy
        (`karume.verify.STORAGE_DTYPES`). If a storage dtype that passes as IR were missing from
        the label vocabulary, `verify_dist` would fail **the moment a distribution declaring that
        dtype is assembled** (the mismatch that actually happened with ADR 0097's `i2`).
        """
        assert set(verify.STORAGE_DTYPES) <= STORAGE_DTYPE_LABELS


class TestStorageOfF16Steps:
    """The f16 steps of `int4-sym-g-f16` are the `i4f16` layout's own, not an f16 weight
    ({@link karume.dist.storage_dtypes} — ADR 0145 decision 3): a role that forbids an f16 series
    (the q4_0 QAT line's model role) still accepts a container whose experts are `i4f16`."""

    def test_the_steps_add_no_f16_to_the_storage(self, tmp_path: Path) -> None:
        (path,) = write_f16_steps_container(tmp_path / "model.krm")

        assert dist.storage_dtypes(path) == {"i4f16"}

    def test_an_f16_prohibition_passes_and_an_i4f16_one_still_fails(self, tmp_path: Path) -> None:
        (path,) = write_f16_steps_container(tmp_path / "model.krm")

        dist.assert_storage_absent("model", path, {"model": ("f16",)})
        with pytest.raises(DistError, match="has i4f16"):
            dist.assert_storage_absent("model", path, {"model": ("i4f16",)})


class TestPlanGates:
    """Checks decided by `plans` alone run **before the first byte is written** (the MUST at the
    top of dist.py).

    Failing after laying out several GB would leave a half-finished tree without a `karume.json`,
    or with the previous one.
    """

    def test_it_refuses_too_many_models_before_writing_anything(self, tmp_path: Path) -> None:
        plans = [
            _synthetic_plan(tmp_path / "series", f"m{index}", "w/model.krm", f"w{index}")
            for index in range(33)
        ]
        out_dir = tmp_path / "models" / "too-many"
        with pytest.raises(DistError, match="models has 33 entries"):
            assemble_family(plans, out_dir, "m0")
        assert not out_dir.exists()

    def test_it_refuses_a_rel_path_that_climbs_out_of_the_model_subtree(
        self, tmp_path: Path
    ) -> None:
        """`A/../../victim` would truncate an existing file outside staging.

        Not a single post-write check would fire — the manifest's root check looks only at the
        leading segment (`A`), and the undeclared-file check looks only inside staging.
        `out_dir / rel_path` matches the declaration in both `is_file()` and size, so
        `verify_dist` passes green to the end.
        """
        victim = tmp_path / "models" / "victim.bin"
        victim.parent.mkdir(parents=True)
        victim.write_bytes(b"someone-elses-bytes")
        plans = [_synthetic_plan(tmp_path / "series", "A", "../../victim.bin", "w")]
        out_dir = tmp_path / "models" / "escape"

        with pytest.raises(DistError, match="a segment with a leading dot"):
            assemble_family(plans, out_dir, "A")

        assert victim.read_bytes() == b"someone-elses-bytes"
        assert not out_dir.exists()

    def test_it_refuses_a_model_name_that_is_not_a_path_segment(self, tmp_path: Path) -> None:
        """The name check used to exist only on the recipe side (the core's assembly path let names
        through)."""
        plans = [_synthetic_plan(tmp_path / "series", "../escape", "w/model.krm", "w")]
        out_dir = tmp_path / "models" / "bad-name"

        with pytest.raises(DistError, match="allowed characters"):
            assemble_family(plans, out_dir, "../escape")
        assert not out_dir.exists()

    def test_it_refuses_two_roles_of_one_model_claiming_the_same_path(self, tmp_path: Path) -> None:
        """A `rel_path` collision within one model would silently let the later one win (the same
        MUST exists across models).

        The later role overwrites the earlier role's data and the manifest declares **both roles
        with the later digest**, so files and table still agree while the `encoder` session reads
        the `decoder` graph.
        """
        series = tmp_path / "series"
        plan = ModelPlan(
            name="A",
            pipeline="anima/1",
            artifacts={
                "encoder": Artifact(
                    rel_path="model.krm", source=_synthetic_series(series, "encoder", "enc")
                ),
                "decoder": Artifact(
                    rel_path="model.krm", source=_synthetic_series(series, "decoder", "dec")
                ),
            },
            weights={
                "enc": {"f16": WeightFiles(file="encoder")},
                "dec": {"f16": WeightFiles(file="decoder")},
            },
            assets={},
            quants={"f16": {"weights": {"enc": "f16", "dec": "f16"}, "session": {}}},
            default_quant="f16",
            pipeline_config={},
        )
        out_dir = tmp_path / "models" / "collided"

        with pytest.raises(DistError, match="is claimed by both"):
            assemble_family([plan], out_dir, "A")
        assert not out_dir.exists()


#: The shared table of the accepted set of a quant's presentation fields. The source of truth is
#: the hub's parser (ruling of 2026-10-11), and the TS side
#: (packages/hub/tests/quant_presentation_test.ts) also runs on **this one file**.
QUANT_PRESENTATION_TABLE = json.loads(
    (REPO_ROOT / "packages" / "hub" / "tests" / "fixtures" / "quant-presentation.json").read_text(
        encoding="utf-8"
    )
)


class TestQuantPresentation:
    """A quant's presentation fields (`label` / `description` of ADR 0075 decision 1) — optional,
    with limits.

    The source of truth for the accepted set is the hub's parser (ruling of 2026-10-11). Looser
    than the hub gives "a distribution that assembles but cannot be read", stricter than the hub
    cannot write values the hub can read — the shared table turns either mismatch red on both
    sides.
    """

    def test_the_limits_are_the_ones_of_the_shared_table(self) -> None:
        assert QUANT_PRESENTATION_TABLE["limits"] == {
            "label": MAX_QUANT_LABEL_CHARS,
            "description": MAX_QUANT_DESCRIPTION_CHARS,
        }

    @pytest.mark.parametrize(
        "case",
        [case for case in QUANT_PRESENTATION_TABLE["cases"] if case["accepted"]],
        ids=lambda case: case["name"],
    )
    def test_it_accepts_what_the_hub_accepts(self, case: Mapping[str, Any]) -> None:
        quant = {case["field"]: case["value"]} if "value" in case else {}
        assert_quant_presentation("m.quants.f16", {"weights": {}, "session": {}, **quant})

    @pytest.mark.parametrize(
        "case",
        [case for case in QUANT_PRESENTATION_TABLE["cases"] if not case["accepted"]],
        ids=lambda case: case["name"],
    )
    def test_it_refuses_what_the_hub_refuses(self, case: Mapping[str, Any]) -> None:
        with pytest.raises(DistError, match=rf"m\.quants\.f16\.{case['field']}"):
            assert_quant_presentation("m.quants.f16", {case["field"]: case["value"]})

    def test_a_seat_may_leave_both_fields_out(self) -> None:
        assert_quant_presentation("m.quants.f16", {"weights": {}, "session": {}})

    @pytest.mark.parametrize(
        ("key", "limit"),
        [("label", MAX_QUANT_LABEL_CHARS), ("description", MAX_QUANT_DESCRIPTION_CHARS)],
    )
    def test_it_accepts_the_longest_value_the_hub_reads(self, key: str, limit: int) -> None:
        assert_quant_presentation("m.quants.f16", {key: "x" * limit})

    @pytest.mark.parametrize(
        ("key", "limit"),
        [("label", MAX_QUANT_LABEL_CHARS), ("description", MAX_QUANT_DESCRIPTION_CHARS)],
    )
    def test_it_refuses_one_character_beyond_the_limit(self, key: str, limit: int) -> None:
        with pytest.raises(DistError, match=f"over the limit of {limit}"):
            assert_quant_presentation("m.quants.f16", {key: "x" * (limit + 1)})

    def test_the_plan_gate_catches_it_before_writing_anything(self, tmp_path: Path) -> None:
        """The check at the plan step (the same place as the size limits of ADR 0041 §7) — no
        failing after laying out several GB."""
        plan = _synthetic_plan(tmp_path / "series", "A", "w/model.krm", "w")
        oversized = {
            "f16": {**plan.quants["f16"], "label": "x" * (MAX_QUANT_LABEL_CHARS + 1)},
        }
        out_dir = tmp_path / "models" / "loud"

        with pytest.raises(DistError, match=r"A\.quants\.f16\.label"):
            assemble_family([replace(plan, quants=oversized)], out_dir, "A")
        assert not out_dir.exists()


class TestManifestJsonLiterals:
    """Only **standard JSON** goes into the manifest — the hub reads it with the browser's
    `JSON.parse`.

    Python's `json` writes `NaN` / `Infinity` by default and reads them back by default (the
    read-back of `verify_dist` lets them through too), so unless the writing side fails, the
    distribution breaks only after shipping, on the user's machine. The same MUST already exists on
    the IR side (`karume.ir.IrGraph.to_json`).
    """

    def test_it_refuses_a_non_finite_float_in_the_manifest(self) -> None:
        with pytest.raises(ValueError, match="JSON compliant"):
            manifest_text({"format": MANIFEST_FORMAT, "models": {"a": {"scale": float("inf")}}})

    def test_it_writes_a_finite_manifest_with_a_trailing_newline(self) -> None:
        assert manifest_text({"format": MANIFEST_FORMAT}).endswith('"karume/5"\n}\n')

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_it_refuses_a_non_finite_preprocessor_channel(self, bad: float) -> None:
        """`<= 0` is false for `NaN`, so a positivity check alone lets it slip through even
        `image_std`."""
        with pytest.raises(DistError, match="non-finite"):
            preprocessor_channels(
                {"image_std": [0.5, bad, 0.5]},
                "image_std",
                "preprocessor_config.json",
                channels=3,
                positive=True,
            )

    def test_it_still_reads_the_finite_channels(self) -> None:
        assert preprocessor_channels(
            {"image_mean": [0.5, 0.25, 1]},
            "image_mean",
            "preprocessor_config.json",
            channels=3,
            positive=False,
        ) == [0.5, 0.25, 1.0]


class TestFamilyAssembly:
    """Several models in one repository (ADR 0041 §2) + shared files only once under `shared/`
    (§5)."""

    def test_it_never_folds_a_path_that_two_byte_strings_both_claim(self, tmp_path: Path) -> None:
        """When two foldable groups share the same relative path, neither is folded (there is only
        one slot per path).

        Folding would let the later group overwrite the earlier group's `shared/` data, giving a
        distribution where the earlier group's models read **different bytes** from the sha256
        their manifest declares. The lengths are equal, so both the size comparison of
        `verify_dist` and the three-field comparison of duplicate paths stay green = a silent
        path.
        """
        rel_path = "text_encoder/model.krm"
        first, second = "weights-D1", "weights-D2"
        assert [len(item) for item in ir_container(mark=first)] == [
            len(item) for item in ir_container(mark=second)
        ]
        plans = [
            _synthetic_plan(tmp_path / "series", name, rel_path, mark)
            for name, mark in (("A", first), ("B", first), ("C", second), ("D", second))
        ]
        out_dir = tmp_path / "models" / "contested"
        manifest = assemble_family(plans, out_dir, "A")

        expected: list[str] = []
        for name, mark in (("A", first), ("B", first), ("C", second), ("D", second)):
            refs = manifest["models"][name]["weights"]["w"]["f16"]["container"]["parts"]
            assert refs == _part_refs(name, rel_path, mark), name
            for ref, payload in zip(refs, ir_container(mark=mark), strict=True):
                assert (out_dir / ref["path"]).read_bytes() == payload, name
            expected.extend(ref["path"] for ref in refs)
        assert not (out_dir / SHARED_DIRNAME).exists()
        assert sorted(verify_dist(out_dir)) == sorted(expected)

    @staticmethod
    def _record_source_digests(monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Records `(model name, relative path)` of the entries whose source hash was taken.

        "Not read" cannot be observed by laying out the files alone (the resulting tree is the
        same whether read or not), so the only read path the sharing check uses is instrumented
        and counted.
        """
        real = dist._source_digest
        taken: list[str] = []

        def recording(artifact: Artifact, memo: dict[Path, str]) -> str:
            taken.append(artifact.rel_path)
            return real(artifact, memo)

        monkeypatch.setattr("karume.dist._source_digest", recording)
        return taken

    def test_it_settles_a_path_two_sizes_claim_without_reading_either(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even at the same relative path, a different size always means different content — not
        shared is decided before taking a hash."""
        rel_path = "text_encoder/model.krm"
        short, long = "weights-A", "a-much-longer-mark"
        # Entries are per part, so the pre-filter applies only if **every part** differs in size
        # from its counterpart.
        assert all(
            len(first) != len(second)
            for first, second in zip(ir_container(mark=short), ir_container(mark=long), strict=True)
        )
        taken = self._record_source_digests(monkeypatch)
        plans = [
            _synthetic_plan(tmp_path / "series", name, rel_path, mark)
            for name, mark in (("A", short), ("B", long))
        ]
        out_dir = tmp_path / "models" / "sized"
        manifest = assemble_family(plans, out_dir, "A")

        assert taken == []
        for name, mark in (("A", short), ("B", long)):
            refs = manifest["models"][name]["weights"]["w"]["f16"]["container"]["parts"]
            assert [ref["path"] for ref in refs] == _part_paths(name, rel_path, mark), name
            for ref, payload in zip(refs, ir_container(mark=mark), strict=True):
                assert (out_dir / ref["path"]).read_bytes() == payload, name
        assert not (out_dir / SHARED_DIRNAME).exists()

    def test_it_still_reads_the_source_to_separate_two_equal_sized_byte_strings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same size with different content cannot be ruled out by the pre-filter — the source
        sha256 tells them apart as before."""
        rel_path = "text_encoder/model.krm"
        first, second = "weights-D1", "weights-D2"
        assert [len(item) for item in ir_container(mark=first)] == [
            len(item) for item in ir_container(mark=second)
        ]
        taken = self._record_source_digests(monkeypatch)
        plans = [
            _synthetic_plan(tmp_path / "series", name, rel_path, mark)
            for name, mark in (("A", first), ("B", second))
        ]
        out_dir = tmp_path / "models" / "equal-sized"
        manifest = assemble_family(plans, out_dir, "A")

        # Entries are per part (2 models × 2 parts) — every entry has a same-sized counterpart, so
        # all of them are read.
        assert taken == _part_rel_paths(rel_path, first) * 2
        for name, mark in (("A", first), ("B", second)):
            refs = manifest["models"][name]["weights"]["w"]["f16"]["container"]["parts"]
            assert [ref["path"] for ref in refs] == _part_paths(name, rel_path, mark), name
            for ref, payload in zip(refs, ir_container(mark=mark), strict=True):
                assert (out_dir / ref["path"]).read_bytes() == payload, name
        assert not (out_dir / SHARED_DIRNAME).exists()

    def test_it_folds_the_matching_pair_while_a_differently_sized_seat_looks_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pre-filter never touches **a foldable group** — only the differently sized loner
        drops out.

        A form where "a foldable group" and "a differently sized loner" share the same relative
        path is the only form whose result differs from a pre-filter applied per relative path (=
        the group dropping out along with it).
        """
        rel_path = "text_encoder/model.krm"
        shared_mark, odd = "same-bytes", "a-longer-byte-string"
        assert all(
            len(first) != len(second)
            for first, second in zip(
                ir_container(mark=shared_mark), ir_container(mark=odd), strict=True
            )
        )
        taken = self._record_source_digests(monkeypatch)
        plans = [
            _synthetic_plan(tmp_path / "series", name, rel_path, mark)
            for name, mark in (("A", shared_mark), ("B", shared_mark), ("C", odd))
        ]
        out_dir = tmp_path / "models" / "mixed"
        manifest = assemble_family(plans, out_dir, "A")

        assert taken == _part_rel_paths(rel_path, shared_mark) * 2
        shared_refs = _part_refs(SHARED_DIRNAME, rel_path, shared_mark)
        for name in ("A", "B"):
            assert (
                manifest["models"][name]["weights"]["w"]["f16"]["container"]["parts"] == shared_refs
            ), name
        loner_refs = _part_refs("C", rel_path, odd)
        assert manifest["models"]["C"]["weights"]["w"]["f16"]["container"]["parts"] == loner_refs
        for ref, payload in zip(loner_refs, ir_container(mark=odd), strict=True):
            assert (out_dir / ref["path"]).read_bytes() == payload
        for ref, payload in zip(shared_refs, ir_container(mark=shared_mark), strict=True):
            assert (out_dir / ref["path"]).read_bytes() == payload
        assert sorted(verify_dist(out_dir)) == sorted(
            ref["path"] for ref in (*shared_refs, *loner_refs)
        )

    def test_it_refuses_a_shared_copy_that_did_not_land_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A shared entry is placed only once, so a broken copy makes **all models alike** point at
        broken data.

        The declared sha256 is taken from the placed file, so the broken bytes and the
        declaration stay in agreement and `verify_dist` stays silent (it does not recompute).
        Only the placement side, which already reads the source for the sharing check, can compare
        against the source sha256.
        """

        def corrupt(artifact: Artifact, dest: Path) -> None:
            materialize(artifact, dest)
            if SHARED_DIRNAME in dest.parts:
                dest.write_bytes(b"corrupted")

        monkeypatch.setattr("karume.dist.materialize", corrupt)
        plans = [
            _synthetic_plan(tmp_path / "series", name, "w/model.krm", "same-bytes")
            for name in ("A", "B")
        ]
        with pytest.raises(DistError, match="disagrees with the source"):
            assemble_family(plans, tmp_path / "models" / "family", "A")

    @pytest.mark.parametrize("names", [("A",), ("A", "B")], ids=["own-seat", "shared-seat"])
    def test_it_refuses_a_series_part_that_disagrees_with_the_descriptor(
        self, tmp_path: Path, names: tuple[str, ...]
    ) -> None:
        """If a series part is broken while keeping its length, the placed file is compared with the
        descriptor's declaration and fails.

        The manifest's sha256 is taken from the placed file, so the broken bytes and the
        declaration land in agreement. Fetching through HF does not apply the block sha256, so no
        check finds this after publication. Even for a shared entry, the comparison with the source
        cannot catch it (the source itself is broken).
        """
        plans = [
            _synthetic_plan(tmp_path / "series", name, "w/model.krm", "same-bytes")
            for name in names
        ]
        total = len(ir_container(mark="same-bytes"))
        weights_part = tmp_path / "series" / "same-bytes" / numbered_name("model.krm", total, total)
        raw = bytearray(weights_part.read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        weights_part.write_bytes(bytes(raw))
        out_dir = tmp_path / "models" / "family"

        with pytest.raises(
            DistError,
            match=rf"the sha256 of part {total - 1} disagrees with the descriptor's declaration",
        ):
            assemble_family(plans, out_dir, "A")

        assert not out_dir.exists()


class TestRootFiles:
    """Legal text at the root of the distribution repository (files the upstream license requires
    as a condition of redistribution).

    The manifest does not declare them (they belong to no model and apply to the distribution
    repository itself), so they are the entries let through by name as exceptions to the
    undeclared-file check.
    """

    def test_it_writes_them_at_the_repository_root_and_verifies(self, tmp_path: Path) -> None:
        """They are placed at the root in UTF-8 and pass `verify_dist` in that state."""
        out_dir = tmp_path / "models" / "legal"
        texts = {"LICENSE.md": "ライセンス原文\n", "NOTICE.md": "notice\n"}
        assemble_family(
            [_synthetic_plan(tmp_path / "series", "A", "w/model.krm", "w")],
            out_dir,
            "A",
            root_files=texts,
        )

        for name, text in texts.items():
            assert (out_dir / name).read_text(encoding="utf-8") == text
        # The manifest declares none of them (they are not assets), yet the check passes.
        assert sorted(verify_dist(out_dir)) == _part_paths("A", "w/model.krm", "w")

    def test_it_refuses_a_name_outside_the_legal_seat_before_writing_anything(
        self, tmp_path: Path
    ) -> None:
        """Only the two legal-text entries — the root is a place the "no undeclared files" net does
        not reach."""
        out_dir = tmp_path / "models" / "smuggled"
        with pytest.raises(DistError, match="entries for legal text only"):
            assemble_family(
                [_synthetic_plan(tmp_path / "series", "A", "w/model.krm", "w")],
                out_dir,
                "A",
                root_files={"LICENSE.md": "ok", "config.json": "{}"},
            )
        assert not out_dir.exists()

    def test_a_family_without_them_still_refuses_an_undeclared_root_file(
        self, tmp_path: Path
    ) -> None:
        """The exception applies only to **those two names** — the net of a family that does not use
        the entries is not loosened."""
        out_dir = tmp_path / "models" / "plain"
        plan = _synthetic_plan(tmp_path / "series", "A", "w/model.krm", "w")
        assemble_family([plan], out_dir, "A")
        (out_dir / "leftover.safetensors").write_bytes(b"stale")

        with pytest.raises(DistError, match=r"leftover\.safetensors"):
            verify_dist(out_dir)

    def test_the_exemption_does_not_reach_a_subdirectory(self, tmp_path: Path) -> None:
        """The exception is by **relative path**, not by name — a file of the same name slipped into
        a model subtree fails as before."""
        out_dir = tmp_path / "models" / "nested"
        assemble_family(
            [_synthetic_plan(tmp_path / "series", "A", "w/model.krm", "w")],
            out_dir,
            "A",
            root_files={"NOTICE.md": "notice\n"},
        )
        (out_dir / "A" / "NOTICE.md").write_text("notice\n", encoding="utf-8")

        with pytest.raises(DistError, match=r"A/NOTICE\.md"):
            verify_dist(out_dir)


class TestAtomicReplacement:
    """Assembly builds in staging and swaps in with a rename — the existing distribution never shows
    a half-done state."""

    def _snapshot(self, out_dir: Path) -> dict[str, bytes]:
        return {rel_path: (out_dir / rel_path).read_bytes() for rel_path in _present(out_dir)}

    def _siblings(self, out_dir: Path) -> list[str]:
        """Working directories left next to the destination (staging / the set-aside copy) — empty
        on success and on failure."""
        return sorted(path.name for path in out_dir.parent.iterdir() if path.name != out_dir.name)

    def _fail_at(self, monkeypatch: pytest.MonkeyPatch, nth: int) -> None:
        """Makes only the `nth` placement an I/O fault (injects a failure partway through several
        GB)."""
        calls = 0

        def failing(artifact: Artifact, dest: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == nth:
                raise OSError("failed partway through placement")
            materialize(artifact, dest)

        monkeypatch.setattr("karume.dist.materialize", failing)

    def _fail_replace_at(self, monkeypatch: pytest.MonkeyPatch, nth: int) -> None:
        """Makes only the `nth` `os.replace` an I/O fault (injects a failure partway through the
        swap)."""
        real = os.replace
        calls = 0

        def failing(src: Path, dst: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == nth:
                raise OSError("failed partway through the swap")
            real(src, dst)

        monkeypatch.setattr("karume.artifacts.os.replace", failing)

    def _versioned_plans(self, root: Path, version: str) -> list[ModelPlan]:
        """The smallest plans of 3 models × 1 role. Each version changes **only the content** (the
        lengths are the same).

        Reassembling with the same content would make a partially overwritten tree
        indistinguishable byte for byte from the previous one (the invariance claim would test
        nothing) — observing fault injection needs a difference between versions.
        """
        return [
            _synthetic_plan(root, name, "w/model.krm", f"{version}-{name}")
            for name in ("A", "B", "C")
        ]

    def test_a_failure_midway_leaves_the_previous_distribution_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Back when updates were in place, this left "old manifest + a tree mixing old and new"."""
        out_dir = tmp_path / "models" / "synthetic"
        assemble_family(self._versioned_plans(tmp_path / "series", "v1"), out_dir, "A")
        before = self._snapshot(out_dir)

        self._fail_at(monkeypatch, nth=2)
        with pytest.raises(OSError, match="failed partway through placement"):
            assemble_family(self._versioned_plans(tmp_path / "series", "v2"), out_dir, "A")

        assert self._snapshot(out_dir) == before
        assert self._siblings(out_dir) == []

    def test_a_failing_card_renderer_leaves_the_previous_distribution_byte_identical(
        self, tmp_path: Path
    ) -> None:
        """The card is inside the swap too — if it cannot be rendered, the swap does not happen at
        all."""
        out_dir = tmp_path / "models" / "synthetic"
        assemble_family(self._versioned_plans(tmp_path / "series", "v1"), out_dir, "A")
        before = self._snapshot(out_dir)

        def explode(manifest: Mapping[str, Any], host_assets: Mapping[FileRefKey, int]) -> str:
            raise DistError("cannot render the card")

        with pytest.raises(DistError, match="cannot render the card"):
            assemble_family(
                self._versioned_plans(tmp_path / "series", "v2"),
                out_dir,
                "A",
                render_card=explode,
            )

        assert self._snapshot(out_dir) == before
        assert self._siblings(out_dir) == []

    def test_a_failing_swap_puts_the_previous_distribution_back_at_the_canonical_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even when the second step of the swap (staging → destination) fails, the distribution
        does not disappear from the canonical path."""
        out_dir = tmp_path / "models" / "synthetic"
        assemble_family(self._versioned_plans(tmp_path / "series", "v1"), out_dir, "A")
        before = self._snapshot(out_dir)

        # MUST: the series is written **before the injection** — `os.replace` is shared by the
        # modules, so catching the series-side swap (`karume.publish`) too would make the subject
        # something other than the assembly's swap.
        plans = self._versioned_plans(tmp_path / "series", "v2")
        self._fail_replace_at(monkeypatch, nth=2)
        with pytest.raises(DistError, match=r"swap \(rename\) into") as failure:
            assemble_family(plans, out_dir, "A")

        # The cause (the I/O fault) stays chained — a swap failure is not mistaken for an
        # assembly failure.
        assert isinstance(failure.value.__cause__, OSError)
        assert self._snapshot(out_dir) == before
        assert self._siblings(out_dir) == []

    def test_a_failing_evacuation_leaves_the_previous_distribution_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even when the first step of the swap (destination → set-aside) fails, os.replace is
        atomic, so the destination stays intact."""
        out_dir = tmp_path / "models" / "synthetic"
        assemble_family(self._versioned_plans(tmp_path / "series", "v1"), out_dir, "A")
        before = self._snapshot(out_dir)

        plans = self._versioned_plans(tmp_path / "series", "v2")
        self._fail_replace_at(monkeypatch, nth=1)
        with pytest.raises(DistError, match=r"moving .* aside") as failure:
            assemble_family(plans, out_dir, "A")

        # The cause (the I/O fault) stays chained — a set-aside failure is not mistaken for an
        # assembly failure.
        assert isinstance(failure.value.__cause__, OSError)
        assert self._snapshot(out_dir) == before
        assert self._siblings(out_dir) == []

    def test_it_restores_a_distribution_left_only_in_the_superseded_slot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case where the previous run failed **between** the two renames — the last-known-good
        in the set-aside slot is restored rather than discarded.

        Even if the next assembly fails, the healthy distribution is back at the canonical path
        (discarding it would stop at "neither destination nor set-aside copy" = the local
        distribution completely gone).
        """
        out_dir = tmp_path / "models" / "synthetic"
        assemble_family(self._versioned_plans(tmp_path / "series", "v1"), out_dir, "A")
        before = self._snapshot(out_dir)
        os.replace(out_dir, out_dir.with_name(out_dir.name + SUPERSEDED_SUFFIX))

        self._fail_at(monkeypatch, nth=2)
        with pytest.raises(OSError, match="failed partway through placement"):
            assemble_family(self._versioned_plans(tmp_path / "series", "v2"), out_dir, "A")

        assert self._snapshot(out_dir) == before
        assert self._siblings(out_dir) == []


class TestTheGraphNameMatchesTheWeightsKey:
    """The container's graph name == the weights key of its entry (container-v1 §2.1).

    The runtime looks it up **by name** with `prepareContainer(opened, <weights key>)`, so a
    container with a diverging spelling would land together with its manifest and only turn into
    "the container has no graph" at the user's `createSession`. The writer-side spelling check
    (comparing the recipe's constants with the AST) looks only at the source, so this is the only
    place comparing the files with the declaration.
    """

    @staticmethod
    def _plan(series: Path, rel_path: str = "w/model.krm") -> ModelPlan:
        return ModelPlan(
            name="A",
            pipeline="anima/1",
            artifacts={"w": Artifact(rel_path=rel_path, source=series)},
            weights={"w": {"f16": WeightFiles(file="w")}},
            assets={},
            quants={"f16": {"weights": {"w": "f16"}, "session": {}}},
            default_quant="f16",
            pipeline_config={},
        )

    def test_a_container_named_after_the_series_directory_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        """A container named after its location (the series name) is not shipped — the reason this
        check exists."""
        series = _write_series(
            tmp_path / "series" / "w", ir_container(mark="w", named="siglip2-so400m-patch14-384")
        )
        out_dir = tmp_path / "models" / "named-after-the-directory"

        with pytest.raises(DistError, match="the container's graph names"):
            assemble_family([self._plan(series)], out_dir, "A")
        assert not out_dir.exists()

    def test_the_same_series_named_after_the_part_passes(self, tmp_path: Path) -> None:
        """Counterpart (not tautological): the same material named after the component passes."""
        series = _write_series(tmp_path / "series" / "w", ir_container(mark="w", named="w"))
        out_dir = tmp_path / "models" / "named-after-the-part"

        manifest = assemble_family([self._plan(series)], out_dir, "A")

        assert set(manifest["models"]["A"]["weights"]) == {"w"}

    def test_one_container_cannot_fill_two_different_keys(self, tmp_path: Path) -> None:
        """One container holds only one graph, so it cannot claim both of two entries.

        Byte-identical sharing (entries with the same key) is a form that passes, so unless the
        dedupe folds by key as well, this form slips through the check.
        """
        series = _write_series(tmp_path / "series" / "w", ir_container(mark="w", named="w"))
        plan = replace(
            self._plan(series),
            artifacts={
                "w": Artifact(rel_path="w/model.krm", source=series),
                "v": Artifact(rel_path="v/model.krm", source=series),
            },
            weights={"w": {"f16": WeightFiles("w")}, "v": {"f16": WeightFiles("v")}},
            quants={"f16": {"weights": {"w": "f16", "v": "f16"}, "session": {}}},
        )
        out_dir = tmp_path / "models" / "two-keys"

        with pytest.raises(
            DistError, match=r"the container's graph names \['w'\] differ from the weights key 'v'"
        ):
            assemble_family([plan], out_dir, "A")
        assert not out_dir.exists()


class TestExternalComponents:
    """Cross-repository component references (the `repo` / `revision` fields of ADR 0038 §7,
    superseded — now ADR 0109 decision 3) — **opt-in**.

    The only purpose is "not uploading the same bytes to two repositories", and without the option
    the distribution stays fully self-contained
    ({@link test_a_plain_assembly_stays_self_contained}).
    """

    _SHARED = "text-encoder"
    _OWN = "transformer!"
    _REVISION = "0123456789abcdef0123456789abcdef01234567"

    def _plan(self, name: str, series: Path, *, shared: str | None = None) -> ModelPlan:
        """A model with two roles (one byte-identical with another repository, one specific to this
        repository).

        `shared` swaps the content of `text_encoder` (to build a form that disagrees with the
        referenced one).
        """
        return ModelPlan(
            name=name,
            pipeline="anima/1",
            artifacts={
                "text_encoder": Artifact(
                    "text_encoder/model.krm",
                    source=_synthetic_series(series, shared or self._SHARED, "text_encoder"),
                ),
                "transformer": Artifact(
                    "transformer/model.krm",
                    source=_synthetic_series(series, self._OWN, "transformer"),
                ),
            },
            weights={
                "text_encoder": {"f16": WeightFiles("text_encoder")},
                "transformer": {"f16": WeightFiles("transformer")},
            },
            assets={},
            quants={
                "f16": {"weights": {"text_encoder": "f16", "transformer": "f16"}, "session": {}}
            },
            default_quant="f16",
            pipeline_config={},
        )

    def _source_dist(self, tmp_path: Path) -> Path:
        """The referenced source (the already assembled distribution of another repository)."""
        source = tmp_path / "models" / "karume-source"
        assemble_family([self._plan("source", tmp_path / "series")], source, "source")
        return source

    def _components(self, source: Path, **overrides: Any) -> ExternalComponents:
        return ExternalComponents(
            **{
                "repo": "hdae/karume-source",
                "revision": self._REVISION,
                "dist": source,
                "model": "source",
                "roles": ("text_encoder",),
                **overrides,
            }
        )

    def _pinned(self, refs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Turns a list of three-field refs into the form "pinned to the referenced source" (repo /
        revision identical on every element)."""
        return [{"repo": "hdae/karume-source", "revision": self._REVISION, **ref} for ref in refs]

    def test_a_plain_assembly_stays_self_contained(self, tmp_path: Path) -> None:
        """Assembly without the option does not change by a single byte — no reference appears and
        all files are present."""
        out_dir = tmp_path / "models" / "karume-plain"
        manifest = assemble_family([self._plan("plain", tmp_path / "series")], out_dir, "plain")

        assert not any(is_external_ref(ref) for _, ref in dist._declared_refs(manifest))
        assert sorted(verify_dist(out_dir)) == sorted(
            [
                *_part_paths("plain", "text_encoder/model.krm", self._SHARED, "text_encoder"),
                *_part_paths("plain", "transformer/model.krm", self._OWN, "transformer"),
            ]
        )

    def test_it_declares_the_named_role_as_a_pinned_reference(self, tmp_path: Path) -> None:
        source = self._source_dist(tmp_path)
        out_dir = tmp_path / "models" / "karume-borrower"

        manifest = assemble_family(
            [self._plan("borrower", tmp_path / "series")],
            out_dir,
            "borrower",
            external=(self._components(source),),
        )

        entry = manifest["models"]["borrower"]["weights"]["text_encoder"]["f16"]
        assert entry["container"]["parts"] == self._pinned(
            _part_refs("source", "text_encoder/model.krm", self._SHARED, "text_encoder")
        )

    def test_the_referenced_bytes_are_not_stored_here_a_second_time(self, tmp_path: Path) -> None:
        """The very reason references exist — placing the files and also writing the reference would
        gain nothing."""
        source = self._source_dist(tmp_path)
        out_dir = tmp_path / "models" / "karume-borrower"

        assemble_family(
            [self._plan("borrower", tmp_path / "series")],
            out_dir,
            "borrower",
            external=(self._components(source),),
        )

        assert not (out_dir / "borrower" / "text_encoder").exists()
        # Cross-repository references are exempt from the existence check, and only the roles
        # specific to this repository remain as files.
        assert sorted(verify_dist(out_dir)) == _part_paths(
            "borrower", "transformer/model.krm", self._OWN, "transformer"
        )

    @pytest.mark.parametrize("revision", ["main", "v0.4.3", "0123456789abcdef", "A" * 40, "0" * 41])
    def test_it_refuses_a_revision_that_is_not_a_full_commit_sha(
        self, tmp_path: Path, revision: str
    ) -> None:
        """A branch name or a tag is not a pin — if the target moves, size / sha256 disagree with
        the files."""
        with pytest.raises(DistError, match="is not a 40-digit commit sha"):
            self._components(tmp_path, revision=revision)

    def test_it_refuses_a_role_the_source_distribution_does_not_declare(
        self, tmp_path: Path
    ) -> None:
        """What the referenced repository lacks cannot be referenced (the form where **the named
        role** is missing there)."""
        source = tmp_path / "models" / "karume-source"
        # The referenced source is a distribution with only `transformer`.
        assemble_family(
            [_synthetic_plan(tmp_path / "series", "source", "transformer/model.krm", self._OWN)],
            source,
            "source",
        )
        out_dir = tmp_path / "models" / "karume-borrower"

        # The spelling is the part role after expansion (`<representative role>#<number>`) — a
        # reference is resolved per part.
        with pytest.raises(
            DistError, match="The file of role 'text_encoder#1' is not in the referenced dist"
        ):
            assemble_family(
                [self._plan("borrower", tmp_path / "series")],
                out_dir,
                "borrower",
                external=(self._components(source),),
            )
        assert not out_dir.exists()

    def test_a_broken_source_manifest_fails_as_a_dist_error(self, tmp_path: Path) -> None:
        """The referenced source's `karume.json` is read without the form checks, so a missing field
        is also a `DistError`.

        A bare `KeyError` would make "the referenced source is broken" indistinguishable from "a
        bug in the check".
        """
        source = self._source_dist(tmp_path)
        manifest_path = source / MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest["models"]["source"]["weights"]["text_encoder"]["f16"]
        del entry["container"]["parts"][1]["path"]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        out_dir = tmp_path / "models" / "karume-borrower"

        with pytest.raises(
            DistError,
            match=r"models\.source\.weights\.text_encoder\.f16\.container\.parts\[1\]\.path",
        ):
            assemble_family(
                [self._plan("borrower", tmp_path / "series")],
                out_dir,
                "borrower",
                external=(self._components(source),),
            )
        assert not out_dir.exists()

    def test_it_refuses_a_reference_whose_bytes_differ_from_the_local_ones(
        self, tmp_path: Path
    ) -> None:
        """A reference with different content would ship "another model's weights" as one's own.

        Shapes and manifest all stay correct, so cutting the comparison here would be silent.
        """
        source = tmp_path / "models" / "karume-source"
        other = self._plan("source", tmp_path / "series", shared="other-bytes!")
        assemble_family([other], source, "source")
        out_dir = tmp_path / "models" / "karume-borrower"

        with pytest.raises(DistError, match="differs from the bytes this assembly builds"):
            assemble_family(
                [self._plan("borrower", tmp_path / "series")],
                out_dir,
                "borrower",
                external=(self._components(source),),
            )
        assert not out_dir.exists()

    def test_it_applies_one_reference_to_every_model_of_a_family(self, tmp_path: Path) -> None:
        """A cross-repository reference also works in a repository with several models (ADR 0087's
        extra repository has this form).

        The condition is "the files of the named role are byte-identical with the referenced ones
        in every model" — when it holds, the role of the same name in every model declares **the
        same reference**, and those bytes are placed nowhere in this repository.
        """
        source = self._source_dist(tmp_path)
        out_dir = tmp_path / "models" / "karume-family"

        series = tmp_path / "series"
        manifest = assemble_family(
            [self._plan("A", series), self._plan("B", series)],
            out_dir,
            "A",
            external=(self._components(source),),
        )

        expected = self._pinned(
            _part_refs("source", "text_encoder/model.krm", self._SHARED, "text_encoder")
        )
        assert (
            manifest["models"]["A"]["weights"]["text_encoder"]["f16"]["container"]["parts"]
            == expected
        )
        assert (
            manifest["models"]["B"]["weights"]["text_encoder"]["f16"]["container"]["parts"]
            == expected
        )
        # The referenced bytes are in no place of this repository (per-model subtrees, `shared/`).
        assert not (out_dir / "A" / "text_encoder").exists()
        assert not (out_dir / "B" / "text_encoder").exists()
        assert not (out_dir / SHARED_DIRNAME / "text_encoder").exists()
        # The role specific to this repository is byte-identical in both models, so it folds into
        # `shared/` as before.
        assert sorted(verify_dist(out_dir)) == _part_paths(
            SHARED_DIRNAME, "transformer/model.krm", self._OWN, "transformer"
        )

    def test_it_refuses_a_family_where_one_model_holds_different_bytes(
        self, tmp_path: Path
    ) -> None:
        """Only one model's files differ from the referenced ones — what "the same role name points
        at different bytes" looks like.

        Letting it through would ship that model a manifest pointing at **another model's
        weights**.
        """
        source = self._source_dist(tmp_path)
        out_dir = tmp_path / "models" / "karume-family"
        series = tmp_path / "series"
        divergent = self._plan("B", series, shared="other-bytes!")

        with pytest.raises(DistError, match="differs from the bytes this assembly builds"):
            assemble_family(
                [self._plan("A", series), divergent],
                out_dir,
                "A",
                external=(self._components(source),),
            )
        assert not out_dir.exists()

    def test_it_refuses_a_family_whose_models_resolve_to_different_references(
        self, tmp_path: Path
    ) -> None:
        """Even with the same bytes, fails when **the target is not unique**.

        A reference writes the same single one for every model, so a combination that resolves to
        a different path per model cannot be declared at all (the message spells the diverging
        role name and both model names).
        """
        source = tmp_path / "models" / "karume-source"
        series = tmp_path / "series"
        shared_series = _synthetic_series(series, self._SHARED, "text_encoder")
        # A referenced source declaring the same bytes at two relative paths (each borrower hits a
        # different entry). The second entry is **another dtype label of the same weights key** —
        # changing the key would make "the container's graph name ≠ the weights key", and the
        # assembly check ({@link karume.dist.assert_weight_components_verified}) would fail before
        # the reference ambiguity.
        base = self._plan("source", series)
        assemble_family(
            [
                replace(
                    base,
                    artifacts={
                        **base.artifacts,
                        "alt_encoder": Artifact("alt/model.krm", source=shared_series),
                    },
                    weights={
                        **base.weights,
                        "text_encoder": {
                            "f16": WeightFiles("text_encoder"),
                            "i8": WeightFiles("alt_encoder"),
                        },
                    },
                    quants={
                        "f16": {
                            "weights": {"text_encoder": "f16", "transformer": "f16"},
                            "session": {},
                        },
                        "i8": {
                            "weights": {"text_encoder": "i8", "transformer": "f16"},
                            "session": {},
                        },
                    },
                )
            ],
            source,
            "source",
        )
        out_dir = tmp_path / "models" / "karume-family"
        borrower = self._plan("B", series)
        elsewhere = replace(
            borrower,
            artifacts={
                **borrower.artifacts,
                "text_encoder": Artifact("alt/model.krm", source=shared_series),
            },
        )

        with pytest.raises(
            DistError, match=r"reference targets disagree between models.*text_encoder.*'A'.*'B'"
        ):
            assemble_family(
                [self._plan("A", series), elsewhere],
                out_dir,
                "A",
                external=(self._components(source),),
            )
        assert not out_dir.exists()

    def _other_source_dist(self, tmp_path: Path) -> Path:
        """A second referenced repository, holding the bytes of this repository's `transformer`."""
        other = tmp_path / "models" / "karume-other"
        assemble_family([self._plan("other", tmp_path / "series")], other, "other")
        return other

    def test_one_build_references_several_repositories(self, tmp_path: Path) -> None:
        """Each target pins its own roles — `text_encoder` from one repository, `transformer` from
        another — and neither is stored here."""
        source = self._source_dist(tmp_path)
        other = self._other_source_dist(tmp_path)
        out_dir = tmp_path / "models" / "karume-borrower"

        manifest = assemble_family(
            [self._plan("borrower", tmp_path / "series")],
            out_dir,
            "borrower",
            external=(
                self._components(source),
                self._components(
                    other, repo="hdae/karume-other", model="other", roles=("transformer",)
                ),
            ),
        )

        weights = manifest["models"]["borrower"]["weights"]
        assert {ref["repo"] for ref in weights["text_encoder"]["f16"]["container"]["parts"]} == {
            "hdae/karume-source"
        }
        assert {ref["repo"] for ref in weights["transformer"]["f16"]["container"]["parts"]} == {
            "hdae/karume-other"
        }
        assert verify_dist(out_dir) == {}

    def test_it_refuses_a_role_named_by_two_references(self, tmp_path: Path) -> None:
        """One role points at one place — merging two claims would silently keep one of them."""
        source = self._source_dist(tmp_path)
        other = self._other_source_dist(tmp_path)
        out_dir = tmp_path / "models" / "karume-borrower"

        with pytest.raises(
            DistError,
            match="Role 'text_encoder' is named by two cross-repository references"
            r" \(hdae/karume-source and hdae/karume-other\)",
        ):
            assemble_family(
                [self._plan("borrower", tmp_path / "series")],
                out_dir,
                "borrower",
                external=(
                    self._components(source),
                    self._components(other, repo="hdae/karume-other", model="other"),
                ),
            )
        assert not out_dir.exists()

    def test_the_callers_check_sees_every_borrowed_file_before_anything_is_written(
        self, tmp_path: Path
    ) -> None:
        """The check receives each target with the file references resolved for it — the same
        `{repo, revision, path, size, sha256}` the manifest then declares — while the output does
        not exist yet."""
        source = self._source_dist(tmp_path)
        out_dir = tmp_path / "models" / "karume-borrower"
        seen: list[tuple[str, list[Mapping[str, Any]], bool]] = []

        def check(components: ExternalComponents, refs: Sequence[Mapping[str, Any]]) -> None:
            seen.append((components.repo, list(refs), out_dir.exists()))

        manifest = assemble_family(
            [self._plan("borrower", tmp_path / "series")],
            out_dir,
            "borrower",
            external=(self._components(source),),
            check_references=check,
        )

        parts = manifest["models"]["borrower"]["weights"]["text_encoder"]["f16"]["container"]
        assert seen == [("hdae/karume-source", parts["parts"], False)]

    def test_each_target_is_checked_with_its_own_borrowed_files(self, tmp_path: Path) -> None:
        """With several targets, a target's check sees exactly the files pinned to it — the Hub
        check looks each path up in that target's commit, so another target's paths would fail it
        (or, merged the other way, escape it)."""
        source = self._source_dist(tmp_path)
        other = self._other_source_dist(tmp_path)
        out_dir = tmp_path / "models" / "karume-borrower"
        seen: list[tuple[str, list[Mapping[str, Any]]]] = []

        def check(components: ExternalComponents, refs: Sequence[Mapping[str, Any]]) -> None:
            seen.append((components.repo, list(refs)))

        manifest = assemble_family(
            [self._plan("borrower", tmp_path / "series")],
            out_dir,
            "borrower",
            external=(
                self._components(source),
                self._components(
                    other, repo="hdae/karume-other", model="other", roles=("transformer",)
                ),
            ),
            check_references=check,
        )

        weights = manifest["models"]["borrower"]["weights"]
        assert seen == [
            ("hdae/karume-source", weights["text_encoder"]["f16"]["container"]["parts"]),
            ("hdae/karume-other", weights["transformer"]["f16"]["container"]["parts"]),
        ]

    def test_it_refuses_to_build_into_the_referenced_distribution(self, tmp_path: Path) -> None:
        """The wholesale replacement would delete the very files the build borrows — refused
        before a byte is written, the referenced mirror left as it was."""
        source = self._source_dist(tmp_path)
        before = sorted(path.name for path in source.rglob("*"))

        with pytest.raises(DistError, match="is the local distribution of hdae/karume-source"):
            assemble_family(
                [self._plan("borrower", tmp_path / "series")],
                tmp_path / "models" / "." / "karume-source",
                "borrower",
                external=(self._components(source),),
            )
        assert sorted(path.name for path in source.rglob("*")) == before

    def test_a_failing_check_stops_the_build_before_anything_is_written(
        self, tmp_path: Path
    ) -> None:
        source = self._source_dist(tmp_path)
        out_dir = tmp_path / "models" / "karume-borrower"

        def check(components: ExternalComponents, refs: Sequence[Mapping[str, Any]]) -> None:
            raise DistError(f"{components.repo} does not hold {len(refs)} files")

        with pytest.raises(DistError, match="hdae/karume-source does not hold"):
            assemble_family(
                [self._plan("borrower", tmp_path / "series")],
                out_dir,
                "borrower",
                external=(self._components(source),),
                check_references=check,
            )
        assert not out_dir.exists()


class TestDeclaredReferences:
    """Cross-repository references declared by the pipeline ({@link Pipeline.references}) and
    pinned per build (`--ref-revision NAME=SHA`) — ADR 0038 §7 addendum 2026-10-28.

    The synthetic pipeline builds the one role `w` from `--series`; the referenced repository
    `karume-source` is assembled from the same series, so its bytes are identical.
    """

    _REVISION = "0123456789abcdef0123456789abcdef01234567"
    _REFERENCE = CrossRepoReference("karume-source", "m", ("w",))

    def _pipeline(self, *references: CrossRepoReference) -> Pipeline:
        return replace(_ONE_PROFILE_PIPELINE, references=lambda _model: references)

    def _source(self, tmp_path: Path) -> Path:
        source = tmp_path / "models" / "karume-source"
        assemble_family([_synthetic_plan(tmp_path, "m", "w/model.krm", "w")], source, "m")
        return source

    def _main(self, tmp_path: Path, pipeline: Pipeline, *extra: str, **hooks: Any) -> Path:
        out_dir = tmp_path / "models" / "karume-borrower"
        main(
            ["--pipeline", "solo", "--series", str(tmp_path), "--out", str(out_dir), *extra],
            pipelines={"solo": pipeline},
            referenced_dist=lambda name: tmp_path / "models" / name,
            **hooks,
        )
        return out_dir

    def test_a_pipeline_is_self_contained_unless_it_declares_references(self) -> None:
        assert _ONE_PROFILE_PIPELINE.references("m") == ()

    def test_main_pins_the_declared_roles_at_the_given_revision(self, tmp_path: Path) -> None:
        self._source(tmp_path)

        out_dir = self._main(
            tmp_path,
            self._pipeline(self._REFERENCE),
            "--ref-revision",
            f"karume-source={self._REVISION}",
        )

        manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        refs = manifest["models"]["m"]["weights"]["w"]["f16"]["container"]["parts"]
        assert {(ref["repo"], ref["revision"]) for ref in refs} == {
            ("hdae/karume-source", self._REVISION)
        }
        assert [ref["path"] for ref in refs] == _part_paths("m", "w/model.krm", "w")
        assert verify_dist(out_dir) == {}

    def test_main_hands_the_borrowed_files_to_the_callers_check(self, tmp_path: Path) -> None:
        self._source(tmp_path)
        seen: list[str] = []

        def check(components: ExternalComponents, refs: Sequence[Mapping[str, Any]]) -> None:
            seen.extend(ref["path"] for ref in refs)

        self._main(
            tmp_path,
            self._pipeline(self._REFERENCE),
            "--ref-revision",
            f"karume-source={self._REVISION}",
            check_references=check,
        )

        assert seen == _part_paths("m", "w/model.krm", "w")

    def test_a_declared_reference_without_a_revision_fails_before_anything_is_written(
        self, tmp_path: Path
    ) -> None:
        """It never falls back to a self-contained distribution silently."""
        self._source(tmp_path)

        with pytest.raises(DistError, match=r"references \['karume-source'\] across repositories"):
            self._main(tmp_path, self._pipeline(self._REFERENCE))
        assert not (tmp_path / "models" / "karume-borrower").exists()

    def test_a_declared_role_the_model_does_not_have_fails_before_anything_is_written(
        self, tmp_path: Path
    ) -> None:
        """Role coverage: a declaration left behind when a recipe's table changed fails loudly."""
        self._source(tmp_path)

        with pytest.raises(DistError, match="points at an unknown role: 'nope'"):
            self._main(
                tmp_path,
                self._pipeline(CrossRepoReference("karume-source", "m", ("nope",))),
                "--ref-revision",
                f"karume-source={self._REVISION}",
            )
        assert not (tmp_path / "models" / "karume-borrower").exists()

    def test_a_revision_for_a_repository_the_build_does_not_reference_fails(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(DistError, match=r"does not reference: \['karume-elsewhere'\]"):
            self._main(
                tmp_path,
                _ONE_PROFILE_PIPELINE,
                "--ref-revision",
                f"karume-elsewhere={self._REVISION}",
            )
        assert not (tmp_path / "models" / "karume-borrower").exists()

    @pytest.mark.parametrize("revision", ["main", "0123456789abcdef", "A" * 40])
    def test_a_revision_that_is_not_a_full_commit_sha_fails(
        self, tmp_path: Path, revision: str
    ) -> None:
        with pytest.raises(DistError, match="is not a 40-digit commit sha"):
            self._main(
                tmp_path,
                self._pipeline(self._REFERENCE),
                "--ref-revision",
                f"karume-source={revision}",
            )

    def test_a_reference_to_the_repository_being_built_fails(self, tmp_path: Path) -> None:
        """The hub rejects it only when loading from that HF repository, i.e. after upload."""
        own = CrossRepoReference("karume-solo-m", "m", ("w",))

        with pytest.raises(
            DistError, match=r"points at the repository being built \(hdae/karume-solo-m\)"
        ):
            self._main(
                tmp_path,
                self._pipeline(own),
                "--ref-revision",
                f"karume-solo-m={self._REVISION}",
            )
        assert not (tmp_path / "models" / "karume-borrower").exists()

    def test_the_own_repository_follows_an_explicit_repo_override(self, tmp_path: Path) -> None:
        with pytest.raises(DistError, match="points at the repository being built"):
            self._main(
                tmp_path,
                self._pipeline(self._REFERENCE),
                "--repo",
                "hdae/karume-source",
                "--ref-revision",
                f"karume-source={self._REVISION}",
            )

    def test_the_core_alone_cannot_locate_a_referenced_distribution(self, tmp_path: Path) -> None:
        """Where mirrors live is the caller's knowledge (ADR 0065) — without the hook it fails."""
        with pytest.raises(DistError, match="no location for their local distributions"):
            main(
                [
                    "--pipeline",
                    "solo",
                    "--series",
                    str(tmp_path),
                    "--out",
                    str(tmp_path / "out"),
                    "--ref-revision",
                    f"karume-source={self._REVISION}",
                ],
                pipelines={"solo": self._pipeline(self._REFERENCE)},
            )

    def test_models_of_one_build_must_declare_the_same_references(self) -> None:
        """One build applies one set of references to every model."""
        pipeline = replace(
            _ONE_PROFILE_PIPELINE,
            references=lambda model: (self._REFERENCE,) if model == "a" else (),
        )

        assert declared_references(pipeline, ["a", "a"]) == (self._REFERENCE,)
        with pytest.raises(DistError, match="Models 'a' and 'b' declare different"):
            declared_references(pipeline, ["a", "b"])

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            (("karume x", "m", ("w",)), "does not spell a repository"),
            (("karume-source", "../m", ("w",)), "allowed characters"),
            (("karume-source", "m", ()), "names not a single role"),
            (("karume-source", "m", ("w", "v", "w")), r"names roles twice: \['w'\]"),
        ],
    )
    def test_a_malformed_declaration_fails_where_it_is_written(
        self, arguments: tuple[str, str, tuple[str, ...]], message: str
    ) -> None:
        with pytest.raises(DistError, match=message):
            CrossRepoReference(*arguments)

    def test_the_declaration_names_the_repository_under_the_distribution_owner(self) -> None:
        assert self._REFERENCE.repo == "hdae/karume-source"

    def test_revisions_are_read_as_name_sha_pairs(self) -> None:
        assert parse_reference_revisions(None) == {}
        assert parse_reference_revisions(["a=1", "b=2"]) == {"a": "1", "b": "2"}

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            (["karume-source"], "is not of the form NAME=SHA"),
            (["=abc"], "is not of the form NAME=SHA"),
            (["karume-source="], "is not of the form NAME=SHA"),
            (["a=1", "a=2"], "names 'a' twice"),
        ],
    )
    def test_a_malformed_revision_fails(self, values: list[str], message: str) -> None:
        with pytest.raises(DistError, match=message):
            parse_reference_revisions(values)


#: A synthetic pipeline with a single attribution (a subject that has only "the form of the
#: table" in place of a real family).
#:
#: NOTE: every real recipe is outside the wheel, so **the core does not import a recipe as a
#: subject** (ADR 0065 decision 3 — dependencies go one way, recipe → core). What is observed here
#: is only the core-side rule "how a passed table is handled"; the content of the real tables is
#: checked by each recipe's `tests/test_distribution.py` and the repo driver's
#: `tests/test_dist_driver.py`.
_ONE_PROFILE_PIPELINE = Pipeline(
    default_model="m",
    repo_name=lambda model: f"karume-solo-{model}",
    plan=lambda series_dir, model: _synthetic_plan(series_dir, model, "w/model.krm", "w"),
    card_profiles={"solo": lambda manifest, repo, host_assets: "solo"},
)

#: A synthetic pipeline with two attributions (a separate function object renders each name).
#:
#: NOTE: the real two-profile family (SBV2) has moved out of the wheel, but **the rule that makes
#: the caller choose is the core's responsibility** ({@link resolve_card_renderer}), so the rule is
#: observed with the form of the table alone. That the real table has two is checked by
#: `tools/export-recipes/sbv2/tests/test_distribution.py`.
_TWO_PROFILE_PIPELINE = Pipeline(
    default_model="m",
    repo_name=lambda model: f"karume-{model}",
    plan=lambda series_dir, model: _synthetic_plan(series_dir, model, "w/model.krm", "w"),
    card_profiles={
        "fn": lambda manifest, repo, host_assets: "fn",
        "jvnv": lambda manifest, repo, host_assets: "jvnv",
    },
)

#: A stand-in for the table the caller passes (a synthetic version of what the repo driver
#: passes).
_CALLER_REGISTRY: Mapping[str, Pipeline] = {
    "solo": _ONE_PROFILE_PIPELINE,
    "duo": _TWO_PROFILE_PIPELINE,
}


class TestCli:
    def test_it_has_no_default_pipeline_of_its_own(self) -> None:
        """No default is baked into the core (the caller passes the accepted set — ADR 0065 decision
        2)."""
        args = build_parser().parse_args([])
        assert args.pipeline is None
        assert args.models is None  # resolved in main (to look up the per-pipeline default)

    def test_the_core_table_is_empty(self) -> None:
        """Every per-family recipe is outside the wheel (ADR 0065 stages 3+4 done).

        A non-empty `PIPELINES` would mean the core wheel has taken family knowledge back.
        """
        assert PIPELINES == {}

    def test_it_still_fails_loudly_with_an_empty_table(self) -> None:
        """Even with the default = an empty table the message does not break (it states "required"
        even with zero choices).

        MUST: an empty value does not pass silently — failing here, before touching a single byte
        of the assets, is the only point that detects a call that forgot to pass the table.
        """
        with pytest.raises(DistError, match="--pipeline is required") as error:
            main([])
        assert "choices: " in str(error.value)
        assert "tools/export-recipes/dist.py" in str(error.value)

    @pytest.mark.parametrize("name", ["siglip2", "anima", ""])
    def test_an_empty_table_accepts_no_pipeline_name(self, name: str) -> None:
        """With an empty accepted set **no name** passes (argparse's choices are the empty set)."""
        with pytest.raises(SystemExit) as raised:
            build_parser().parse_args(["--pipeline", name])
        assert raised.value.code == 2

    def test_a_caller_supplied_registry_carries_its_own_default(self) -> None:
        """The repository's dist driver passes the table with the recipes added and a default (its
        accepted set is authoritative)."""
        args = build_parser(_CALLER_REGISTRY, "solo").parse_args([])
        assert args.pipeline == "solo"

    def test_a_caller_supplied_registry_widens_the_accepted_names(self) -> None:
        """Only the names of the passed table pass (independent of the core's empty table)."""
        args = build_parser(_CALLER_REGISTRY, "solo").parse_args(["--pipeline", "duo"])
        assert args.pipeline == "duo"

    def test_the_model_flag_accumulates_into_a_family(self) -> None:
        args = build_parser().parse_args(["--model", "F1", "--model", "F2"])
        assert args.models == ["F1", "F2"]

    def test_it_has_no_place_of_its_own_to_write_into(self) -> None:
        """The default locations are also passed by the caller (`models/` / `outputs/series/` are
        repo topology)."""
        args = build_parser().parse_args([])
        assert args.out is None
        assert args.series is None

    def test_it_refuses_to_assemble_without_a_series_root(self) -> None:
        """MUST: fails when not given — never silently reads some cwd-relative place as the
        series."""
        with pytest.raises(DistError, match="--series is required"):
            main(["--pipeline", "solo"], pipelines=_CALLER_REGISTRY)

    def test_it_refuses_to_assemble_without_an_output_directory(self, tmp_path: Path) -> None:
        """A call that passes no hook for the default `--out` fails here, before touching the
        assets."""
        with pytest.raises(DistError, match="--out is required"):
            main(["--pipeline", "solo", "--series", str(tmp_path)], pipelines=_CALLER_REGISTRY)

    def test_a_caller_supplied_hook_decides_where_the_default_lands(self, tmp_path: Path) -> None:
        """A call that passes the hook goes through without `--out` and assembles into **the place
        the hook returns**.

        The hook receives the resolved pipeline and the list of model names (everything needed to
        spell the repository name).
        """
        seen: list[tuple[Pipeline, list[str]]] = []

        def hook(pipeline: Pipeline, models: Sequence[str]) -> Path:
            seen.append((pipeline, list(models)))
            return tmp_path / "models" / "karume-solo-m"

        main(
            ["--pipeline", "solo", "--series", str(tmp_path)],
            pipelines=_CALLER_REGISTRY,
            default_out_dir=hook,
        )

        assert seen == [(_ONE_PROFILE_PIPELINE, ["m"])]
        assert verify_dist(tmp_path / "models" / "karume-solo-m")


class TestCardProfile:
    """Choosing the attribution profile (`--card-profile`) — a wrong attribution is noticed only
    after shipping."""

    def test_it_is_unset_until_asked_for(self) -> None:
        assert build_parser().parse_args([]).card_profile is None
        assert build_parser().parse_args(["--card-profile", "jvnv"]).card_profile == "jvnv"

    def test_a_pipeline_with_one_profile_needs_no_choice(self) -> None:
        """A pipeline with a single attribution is not asked (there is nothing to choose)."""
        profiles = _ONE_PROFILE_PIPELINE.card_profiles
        assert len(profiles) == 1
        assert resolve_card_renderer(_ONE_PROFILE_PIPELINE, None) is next(iter(profiles.values()))

    def test_it_refuses_to_pick_an_attribution_when_several_exist(self) -> None:
        """Silently picking a default would leave the previous family's attribution on a new
        family."""
        with pytest.raises(DistError, match="--card-profile") as error:
            resolve_card_renderer(_TWO_PROFILE_PIPELINE, None)
        assert "fn" in str(error.value)
        assert "jvnv" in str(error.value)

    def test_it_refuses_a_profile_it_does_not_have(self) -> None:
        with pytest.raises(DistError, match="jvnv"):
            resolve_card_renderer(_TWO_PROFILE_PIPELINE, "FN9")

    def test_it_resolves_each_name_to_its_own_renderer(self) -> None:
        """A separate renderer per name (with a wrong binding, two families would render the same
        card)."""
        profiles = _TWO_PROFILE_PIPELINE.card_profiles
        assert sorted(profiles) == ["fn", "jvnv"]
        assert resolve_card_renderer(_TWO_PROFILE_PIPELINE, "jvnv") is profiles["jvnv"]
        assert profiles["fn"] is not profiles["jvnv"]


def _repo_echo_pipeline(repo_name: Callable[[str], str]) -> Pipeline:
    """A synthetic pipeline that writes **only the passed repo** into the card (a subject for
    observing the repo of the Usage example)."""
    return Pipeline(
        default_model="m",
        repo_name=repo_name,
        plan=lambda series_dir, model: _synthetic_plan(series_dir, model, "w/model.krm", "w"),
        card_profiles={"echo": lambda manifest, repo, host_assets: repo},
    )


#: A pipeline whose repository name contains the model name (one repository per model by default
#: — a bundling name does not follow from the declaration).
_PER_MODEL_REPO_PIPELINE = _repo_echo_pipeline(lambda model: f"karume-voice-{model}")

#: A pipeline where every model declares the same one repository (one repository per family).
_ONE_REPO_PIPELINE = _repo_echo_pipeline(lambda _model: "karume-family")

_REPO_REGISTRY: Mapping[str, Pipeline] = {
    "voice": _PER_MODEL_REPO_PIPELINE,
    "family": _ONE_REPO_PIPELINE,
}


class TestCardRepository:
    """The repo spelled in the card's Usage example — the source of truth is the pipeline's
    declaration, and `--repo` is the explicit override.

    The location (`--out`) is not a fact about the repository. Deriving from it would publish the
    card of a distribution built in a differently named directory still pointing at a repository
    that does not exist.
    """

    def _card(self, tmp_path: Path, out_name: str, *argv: str) -> str:
        out_dir = tmp_path / out_name
        main(
            [*argv, "--series", str(tmp_path / "series"), "--out", str(out_dir)],
            pipelines=_REPO_REGISTRY,
        )
        return (out_dir / MODEL_CARD_FILENAME).read_text(encoding="utf-8")

    def test_the_card_names_the_repository_the_pipeline_declares(self, tmp_path: Path) -> None:
        assert self._card(tmp_path, "karume-voice-m", "--pipeline", "voice") == (
            "hdae/karume-voice-m"
        )

    def test_the_output_directory_name_does_not_reach_the_card(self, tmp_path: Path) -> None:
        """Building under a staging name gives the same card as building under the name the
        procedure uses."""
        staged = self._card(tmp_path, "karume-voice-m-release", "--pipeline", "voice")
        assert staged == self._card(tmp_path, "karume-voice-m", "--pipeline", "voice")
        assert "release" not in staged

    def test_models_that_declare_one_repository_are_assembled_under_it(
        self, tmp_path: Path
    ) -> None:
        card = self._card(tmp_path, "out", "--pipeline", "family", "--model", "a", "--model", "b")
        assert card == "hdae/karume-family"

    def test_it_refuses_to_guess_the_name_of_a_repository_that_bundles_models(
        self, tmp_path: Path
    ) -> None:
        """A bundling whose declarations disagree is not guessed — and it fails **before assembly**
        (leaving no debris)."""
        with pytest.raises(DistError, match="--repo") as error:
            self._card(tmp_path, "out", "--pipeline", "voice", "--model", "a", "--model", "b")
        assert "karume-voice-a" in str(error.value)
        assert "karume-voice-b" in str(error.value)
        assert not (tmp_path / "out").exists()

    def test_an_explicit_repository_overrides_the_declaration(self, tmp_path: Path) -> None:
        argv = ["--pipeline", "voice", "--model", "a", "--model", "b"]
        card = self._card(tmp_path, "out", *argv, "--repo", "hdae/karume-voices")
        assert card == "hdae/karume-voices"

    @pytest.mark.parametrize("repo", ["karume-voices", "hdae/karume/voices", "hdae/", "/voices"])
    def test_it_refuses_a_repository_that_is_not_owner_and_name(self, repo: str) -> None:
        with pytest.raises(DistError, match="<owner>/<name>"):
            resolve_repo(_PER_MODEL_REPO_PIPELINE, ["m"], repo)

    def test_an_override_needs_an_explicit_output_directory(self, tmp_path: Path) -> None:
        """The default destination is spelled with the declared name — overriding the name alone
        would split the location from the card."""
        called: list[Sequence[str]] = []

        def hook(pipeline: Pipeline, models: Sequence[str]) -> Path:
            called.append(models)
            return tmp_path / pipeline.repo_name(models[0])

        with pytest.raises(DistError, match="--out"):
            main(
                ["--pipeline", "voice", "--series", str(tmp_path), "--repo", "hdae/karume-voices"],
                pipelines=_REPO_REGISTRY,
                default_out_dir=hook,
            )
        assert called == []


class TestPartExpansion:
    """A split component is declared as **a part list of several elements** resolved from the
    files.

    How many parts there are depends on the bytes written (`karume.container`), so it cannot be
    written in the pipeline's table — assembly follows the files from the representative path
    ({@link karume.dist.expand_weight_parts}).
    """

    def _plan(self, series: Path, sources: Sequence[bytes]) -> ModelPlan:
        """A plan with one role only. The series side holds the part sequence (head = part 0 with
        the two documents) with numbered names."""
        source = _write_series(series, sources)
        return ModelPlan(
            name="A",
            pipeline="anima/1",
            artifacts={"w": Artifact(rel_path="w/model.krm", source=source)},
            weights={"w": {"f16": WeightFiles(file="w")}},
            assets={},
            quants={"f16": {"weights": {"w": "f16"}, "session": {}}},
            default_quant="f16",
            pipeline_config={},
        )

    def test_a_split_component_is_declared_as_an_ordered_part_list(self, tmp_path: Path) -> None:
        payloads = ir_parts(3, mark="w")
        plan = self._plan(tmp_path / "series", payloads)
        out_dir = tmp_path / "models" / "partitioned"

        manifest = assemble_family([plan], out_dir, "A")

        entry = manifest["models"]["A"]["weights"]["w"]["f16"]
        assert [ref["path"] for ref in entry["container"]["parts"]] == [
            "A/w/model-00001-of-00003.krm",
            "A/w/model-00002-of-00003.krm",
            "A/w/model-00003-of-00003.krm",
        ]
        # The order MUST be by part number (head = part 0 with the two documents). The contents
        # are placed by number too.
        assert [ref["size"] for ref in entry["container"]["parts"]] == [
            len(item) for item in payloads
        ]
        assert [ref["sha256"] for ref in entry["container"]["parts"]] == [
            hashlib.sha256(item).hexdigest() for item in payloads
        ]
        for ref, payload in zip(entry["container"]["parts"], payloads, strict=True):
            assert (out_dir / ref["path"]).read_bytes() == payload

    def test_the_split_parts_are_covered_by_the_declaration_check(self, tmp_path: Path) -> None:
        """The comparison follows the list to the files (they do not fail the undeclared-file
        check)."""
        payloads = ir_parts(3, mark="w")
        plan = self._plan(tmp_path / "series", payloads)
        out_dir = tmp_path / "models" / "partitioned"

        assemble_family([plan], out_dir, "A")

        assert verify_dist(out_dir) == {
            f"A/w/model-{index:05d}-of-{len(payloads):05d}.krm": len(payload)
            for index, payload in enumerate(payloads, start=1)
        }

    def test_the_smallest_component_still_declares_two_elements(self, tmp_path: Path) -> None:
        """Even an asset of a few KB has two or more elements, part 0 (the two documents) + part 1
        (const) (ADR 0109 decision 3)."""
        plan = self._plan(tmp_path / "series", ir_container(mark="w"))
        out_dir = tmp_path / "models" / "whole"

        manifest = assemble_family([plan], out_dir, "A")

        entry = manifest["models"]["A"]["weights"]["w"]["f16"]
        assert [ref["path"] for ref in entry["container"]["parts"]] == [
            "A/w/model-00001-of-00003.krm",
            "A/w/model-00002-of-00003.krm",
            "A/w/model-00003-of-00003.krm",
        ]

    def test_the_leftovers_of_a_previous_export_fail_loudly(self, tmp_path: Path) -> None:
        """A single file next to numbered ones leaves "which one to ship" undecided."""
        series = tmp_path / "series"
        plan = self._plan(series, ir_parts(3, mark="w"))
        (series / "model.krm").write_bytes(b"stale")

        with pytest.raises(DistError, match="exist side by side"):
            assemble_family([plan], tmp_path / "models" / "mixed", "A")

    def test_a_split_component_that_another_seat_also_names_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        """An assets entry is a single-file reference — it cannot point at a role that became
        several parts."""
        plan = self._plan(tmp_path / "series", ir_parts(3, mark="w"))
        plan = replace(plan, assets={"table": "w"})

        with pytest.raises(DistError, match="assets also point at"):
            assemble_family([plan], tmp_path / "models" / "aliased", "A")


class TestRequiredLimits:
    """Assembly derives a quant's `requiredLimits` (the ADR 0038 §7 addendum, superseded — the
    derivation is ADR 0089 decision 3) **from the files** and writes it.

    The check of the derivation rules themselves (what counts as demand, from where values are
    written) is `test_limits.py` — this checks only the wiring: which part of the files is the
    entry point, how it lands in the quant entry, whether a plan that writes the same field itself
    fails. The two tests that swap the defaults do so because synthetic assets (a few KB) do not
    reach the 128MiB band (observing "which part of the files the derivation looks at" without
    building real distribution assets).
    """

    #: The fixture's largest tensor = the 4×32 linear weight in f32 = 512 bytes (`ir_fixtures`).
    _LARGEST_TENSOR: ClassVar[int] = 512

    #: The block limit that splits the 512-byte weight above (128 bytes per row) into **2
    #: pieces** (§4.1).
    _PIECE_BLOCK: ClassVar[int] = 256

    def _plan(
        self,
        series: Path,
        *,
        pipeline_config: Mapping[str, Any] | None = None,
        quant: Mapping[str, Any] | None = None,
        block_bytes: int | None = None,
    ) -> ModelPlan:
        """A plan with one role only (the weights entry is a valid IR container = the entry point of
        the derivation)."""
        source = _write_series(
            series, ir_container(mark="w", block_bytes=block_bytes or BLOCK_MAX_BYTES)
        )
        return ModelPlan(
            name="A",
            pipeline="anima/1",
            artifacts={"w": Artifact(rel_path="w/model.krm", source=source)},
            weights={"w": {"f16": WeightFiles(file="w")}},
            assets={},
            quants={"f16": {"weights": {"w": "f16"}, "session": {}, **(quant or {})}},
            default_quant="f16",
            pipeline_config=dict(pipeline_config or {}),
        )

    def _assemble(self, tmp_path: Path, plan: ModelPlan) -> dict[str, Any]:
        return assemble_family([plan], tmp_path / "models" / "limits", "A")

    def test_a_distribution_within_the_defaults_declares_nothing(self, tmp_path: Path) -> None:
        """Assets that run on default specs get no field (the semantics "no field = runs on the
        defaults")."""
        manifest = self._assemble(tmp_path, self._plan(tmp_path / "series"))

        assert manifest["models"]["A"]["quants"]["f16"] == {
            "weights": {"w": "f16"},
            "session": {},
        }

    def test_it_bakes_the_largest_tensor_of_the_container(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The demand comes from the tensor sizes of **the files that go into the distribution**
        (not a recomputation from the model definition)."""
        monkeypatch.setattr(
            limits,
            "WEBGPU_DEFAULT_LIMITS",
            {"maxBufferSize": 400, "maxStorageBufferBindingSize": 200},
        )

        manifest = self._assemble(tmp_path, self._plan(tmp_path / "series"))

        assert manifest["models"]["A"]["quants"]["f16"]["requiredLimits"] == {
            "maxBufferSize": self._LARGEST_TENSOR,
            "maxStorageBufferBindingSize": self._LARGEST_TENSOR,
        }
        # The written field is also in the placed `karume.json` (not decoration on the return
        # value only).
        written = json.loads(
            (tmp_path / "models" / "limits" / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        assert written["models"]["A"]["quants"]["f16"]["requiredLimits"] == {
            "maxBufferSize": self._LARGEST_TENSOR,
            "maxStorageBufferBindingSize": self._LARGEST_TENSOR,
        }

    def test_a_tensor_split_into_pieces_is_baked_as_its_whole_parent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The demand of a piece list (rule ④ of container-v1 §5) is **the parent's full length**
        (the GPU does not know about pieces).

        The derivation sums the blocks of the merged supply plan (`dist.component_demand_bytes`).
        Degrading to the maximum per piece would give one fragment here (256 bytes) and drop the
        `maxBufferSize` field — the form "satisfies the declaration yet fails in `createSession`".
        The counterpart is {@link test_it_bakes_the_largest_tensor_of_the_container} right above
        (the same series with zero pieces gives the same 512).
        """
        monkeypatch.setattr(
            limits,
            "WEBGPU_DEFAULT_LIMITS",
            {"maxBufferSize": 400, "maxStorageBufferBindingSize": 200},
        )
        out_dir = tmp_path / "models" / "limits"
        plan = self._plan(tmp_path / "series", block_bytes=self._PIECE_BLOCK)

        manifest = self._assemble(tmp_path, plan)

        # The subject really is split — an observation so that the claim does not silently become
        # tautological on a day it is not split.
        entry = manifest["models"]["A"]["weights"]["w"]["f16"]
        parts = [out_dir / ref["path"] for ref in entry["container"]["parts"]]
        supplies = verify.verify_container(parts).graphs["w"].supplies
        assert max(len(supply.blocks) for supply in supplies.values()) == 2
        assert manifest["models"]["A"]["quants"]["f16"]["requiredLimits"] == {
            "maxBufferSize": self._LARGEST_TENSOR,
            "maxStorageBufferBindingSize": self._LARGEST_TENSOR,
        }

    def test_it_declares_only_the_limit_the_demand_actually_exceeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two have different defaults, so in the band that exceeds only one, there is only one
        field (the form of the real irodori f16 distribution)."""
        monkeypatch.setattr(
            limits,
            "WEBGPU_DEFAULT_LIMITS",
            {"maxBufferSize": 4096, "maxStorageBufferBindingSize": 200},
        )

        manifest = self._assemble(tmp_path, self._plan(tmp_path / "series"))

        assert manifest["models"]["A"]["quants"]["f16"]["requiredLimits"] == {
            "maxStorageBufferBindingSize": self._LARGEST_TENSOR
        }

    def test_a_state_slot_can_be_the_largest_resident_buffer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In a series with a large KV capacity **the state slot is larger than the largest
        tensor**.

        A container in states form cannot be synthesized (surgically modified IR = it would have
        to be built with the attention contract), so only the reader is swapped to observe that
        "state counts as demand too". The capacity comes from the `pipelineConfig` field (the
        binding point is `createGenerationContext` — ADR 0066, addendum 2026-08-17 item 7).
        """
        monkeypatch.setattr(
            dist,
            "ir_graph",
            lambda path, verified=None: {
                "states": {"kv": {"dtype": "f32", "shape": [1, 1, "C", 512]}}
            },
        )
        plan = self._plan(tmp_path / "series", pipeline_config={"maxPosition": 200_000})

        manifest = self._assemble(tmp_path, plan)

        slot_bytes = 200_000 * 512 * 4
        assert slot_bytes > self._LARGEST_TENSOR
        assert manifest["models"]["A"]["quants"]["f16"]["requiredLimits"] == {
            "maxBufferSize": slot_bytes,
            "maxStorageBufferBindingSize": slot_bytes,
        }

    def test_a_state_slot_the_config_cannot_bind_fails_loudly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fails without a capacity field — silently leaving the state out would write a "declared,
        yet insufficient" field."""
        monkeypatch.setattr(
            dist,
            "ir_graph",
            lambda path, verified=None: {
                "states": {"kv": {"dtype": "f32", "shape": [1, 1, "C", 512]}}
            },
        )
        plan = self._plan(tmp_path / "series")

        with pytest.raises(DistError, match="maxPosition"):
            self._assemble(tmp_path, plan)

    def test_it_refuses_a_plan_that_writes_the_field_itself(self, tmp_path: Path) -> None:
        """A derivable value is not also held in the table (no form where only the table claims a
        stale lower bound)."""
        plan = self._plan(
            tmp_path / "series", quant={"requiredLimits": {"maxBufferSize": 1_073_741_824}}
        )

        with pytest.raises(DistError, match="requiredLimits"):
            self._assemble(tmp_path, plan)

    def test_nothing_is_written_when_the_derivation_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The derivation runs **before a single byte is written** (a failure leaves not a single
        file of a half-done distribution)."""
        monkeypatch.setattr(
            dist,
            "ir_graph",
            lambda path, verified=None: {
                "states": {"kv": {"dtype": "f32", "shape": [1, 1, "C", 512]}}
            },
        )
        out_dir = tmp_path / "models" / "limits"

        with pytest.raises(DistError):
            assemble_family([self._plan(tmp_path / "series")], out_dir, "A")

        assert not out_dir.exists()


class TestInputContainerVerification:
    """Assembly does not trust input containers as "verified in the past" (CG4-1).

    A series directory is a mutable place rewritten by swaps (the module doc of `dist`), so the
    operational accident of **pointing `--series` at a series written by an old exporter** is a
    real possibility. Assembly itself reads only the storage and part of the IR, so a broken
    descriptor / binding table would slip through the family-specific checks into the
    distribution and fail only at the user's `createSession`.
    """

    def _tampered(self, payloads: Sequence[bytes]) -> list[bytes]:
        """A sequence with the weight parts unchanged and **the graph descriptor of part 0**
        rewritten by hand.

        Exactly a manual tampering / the output of another implementation — moves
        `capabilities.ops` away from the union of `graphs[].requires.ops` (the invariant of
        container-v1 §2.1). Both the undeclared-file check and the size comparison stay green, so
        only the check that reopens the input containers can fail it.
        """
        part0, *rest = payloads
        header = read_header(part0[:HEADER_BYTES])
        begin = HEADER_BYTES
        graph = json.loads(part0[begin : begin + header.graph_length])
        model = part0[begin + header.graph_length :]
        graph["capabilities"]["ops"] = [*graph["capabilities"]["ops"], "linear_v2"]
        rewritten = canonical_json(graph).encode("utf-8")
        head = write_header(
            kind="model", graph_length=len(rewritten), model_length=header.model_length
        )
        return [head + rewritten + model, *rest]

    def _plan(self, series: Path) -> ModelPlan:
        """A model with two entries — only the container of **the entry no quant selects** (`f32`)
        is tampered with.

        MUST: the subject is the unselected entry. For a selected entry the `requiredLimits`
        derivation opens the same container, so another check would fail even without the
        preflight = it would not observe tautology. What only the preflight looks at is "the
        dtype entries no quant selects", and shipping one of them broken is the hole it closes.
        """
        return ModelPlan(
            name="A",
            pipeline="anima/1",
            artifacts={
                "w": Artifact(
                    rel_path="w/model.f16.krm",
                    source=_write_series(
                        series / "sound", ir_container(mark="sound"), "model.f16.krm"
                    ),
                ),
                "broken": Artifact(
                    rel_path="w/model.f32.krm",
                    source=_write_series(
                        series / "broken",
                        self._tampered(ir_container(mark="broken")),
                        "model.f32.krm",
                    ),
                ),
            },
            weights={"w": {"f16": WeightFiles(file="w"), "f32": WeightFiles(file="broken")}},
            assets={},
            quants={"f16": {"weights": {"w": "f16"}, "session": {}}},
            default_quant="f16",
            pipeline_config={},
        )

    def test_a_weights_container_whose_declaration_is_inconsistent_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "models" / "tampered"

        # Checks the layer that fails too — failing in **the container's structural checks**
        # rather than on "file missing" is itself the definition of the hole this check closes
        # (the files are all there, yet the declarations are inconsistent).
        with pytest.raises(
            DistError,
            match=r"does not satisfy the container rules: graphDescriptor\.capabilities\.ops",
        ):
            assemble_family([self._plan(tmp_path / "series")], out_dir, "A")
        assert not out_dir.exists()

    def test_without_the_preflight_the_same_distribution_is_assembled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fault injection — without the check the broken container lands as is (proof that the test
        is not tautological).

        That none of the other checks (declaration-to-file comparison, the undeclared-file check)
        sees this tampering is the very meaning of the test above.
        """
        monkeypatch.setattr(dist, "assert_weight_components_verified", lambda partitioned: {})
        out_dir = tmp_path / "models" / "tampered"

        assemble_family([self._plan(tmp_path / "series")], out_dir, "A")

        assert (out_dir / MANIFEST_FILENAME).is_file()

    def _unknown_op_plan(self, series: Path) -> ModelPlan:
        """A plan whose weights entry points at **a container that declares an op outside the
        vocabulary** (a form that passes the structural checks).

        The subject is an entry a quant selects — the claim of the IR acceptance rules is that
        "every entry's container passes them before landing in the distribution", so observing
        on an unselected entry would put the hole somewhere else.
        """
        source = _write_series(series / "odd", with_an_unknown_op(ir_container(mark="odd")))
        return ModelPlan(
            name="A",
            pipeline="anima/1",
            artifacts={"w": Artifact(rel_path="w/model.f16.krm", source=source)},
            weights={"w": {"f16": WeightFiles(file="w")}},
            assets={},
            quants={"f16": {"weights": {"w": "f16"}, "session": {}}},
            default_quant="f16",
            pipeline_config={},
        )

    def test_a_weights_container_that_declares_an_unknown_op_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        """A container declaring IR that is structurally correct but **the runtime cannot execute**
        is not placed.

        `verify_container` does not touch the op vocabulary, so only the IR acceptance rules
        ({@link karume.verify.assert_ir_accepted}) can fail it — without them, "a container
        declaring an op outside the vocabulary" would land in the distribution and fail only at the
        user's `createSession`.
        """
        out_dir = tmp_path / "models" / "odd"

        with pytest.raises(DistError, match="unsupported ops"):
            assemble_family([self._unknown_op_plan(tmp_path / "series")], out_dir, "A")
        assert not out_dir.exists()

    def test_without_the_preflight_the_unknown_op_container_is_assembled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fault injection — without the check the container with an op outside the vocabulary lands
        as is (proof that the test is not tautological)."""
        monkeypatch.setattr(dist, "assert_weight_components_verified", lambda partitioned: {})
        out_dir = tmp_path / "models" / "odd"

        assemble_family([self._unknown_op_plan(tmp_path / "series")], out_dir, "A")

        assert (out_dir / MANIFEST_FILENAME).is_file()

    def test_a_generated_payload_in_the_weights_seat_fails_loudly(self) -> None:
        """A generated file (`payload`) is not an IR container — put in a weights entry, it fails at
        the plan's entry point.

        Letting it through would declare that role alone as `parts: [one element]` without the
        full IR v1 verification, part resolution or demand derivation (a spelling mistake that
        passes a table / asset role name to weights would go green through every check and fail
        only at the user's `createSession`).
        """
        with pytest.raises(
            DistError, match=r"A\.weights: role 'w' points at a generated file \(payload\)"
        ):
            ModelPlan(
                name="A",
                pipeline="anima/1",
                artifacts={"w": Artifact(rel_path="w/model.krm", payload=b"not-a-container")},
                weights={"w": {"f16": WeightFiles(file="w")}},
                assets={},
                quants={"f16": {"weights": {"w": "f16"}, "session": {}}},
                default_quant="f16",
                pipeline_config={},
            )

    def test_the_same_payload_in_the_assets_seat_is_assembled(self, tmp_path: Path) -> None:
        """Counterpart — `assets` is not an IR container entry, so a generated file passes as is
        (the form the raw rope table, style tables and speaker embeddings actually use)."""
        plan = ModelPlan(
            name="A",
            pipeline="anima/1",
            artifacts={
                "w": Artifact(
                    rel_path="w/model.krm",
                    source=_synthetic_series(tmp_path / "series", "w"),
                ),
                "table": Artifact(rel_path="tables/table.safetensors", payload=b"not-a-container"),
            },
            weights={"w": {"f16": WeightFiles(file="w")}},
            assets={"table": "table"},
            quants={"f16": {"weights": {"w": "f16"}, "session": {}}},
            default_quant="f16",
            pipeline_config={},
        )

        manifest = assemble_family([plan], tmp_path / "models" / "generated", "A")

        assert manifest["models"]["A"]["assets"]["table"] == {
            "path": "A/tables/table.safetensors",
            "size": len(b"not-a-container"),
            "sha256": hashlib.sha256(b"not-a-container").hexdigest(),
        }


class TestExternalPartitionedComponents:
    """**Cross-repository references to a split component** — each element of `container.parts`
    becomes one reference (ADR 0038 §7 / ADR 0071 decision 2, "each element goes through the
    usual FileRef checks as is"; both superseded — now ADR 0109 decision 3, cross-repository
    references per container).

    The actual need is the turbo repository's shared text_encoder (over 1GiB, split into several
    parts); if one role could only write one reference, the very reason references exist, "not
    uploading the same bytes to two repositories", would not reach split assets.
    """

    _REVISION = "0123456789abcdef0123456789abcdef01234567"
    _REPO = "hdae/karume-source"
    #: Uses a container whose part 1 (const region) is **not empty** — a 0-byte part cannot be the
    #: subject of the fault injection "rewrite while keeping the length" (there are no other bytes
    #: of the same length).
    _SHARDS = tuple(ir_container(mark="text-encoder", named="text_encoder"))
    _OWN = "transformer"

    def _series(self, root: Path, payloads: Sequence[bytes]) -> Path:
        """Writes the series output and returns **the representative path** (numbered = the part
        list the files decide)."""
        return _write_series(root, payloads)

    def _plan(self, name: str, series: Path) -> ModelPlan:
        """A model with two roles (`text_encoder` byte-identical with another repository,
        `transformer` specific to this repository).

        The series of the role specific to this repository is written next to the series of
        `text_encoder` (a weights entry is a placement only — `dist.ModelPlan.__post_init__`).
        """
        return ModelPlan(
            name=name,
            pipeline="anima/1",
            artifacts={
                "text_encoder": Artifact("text_encoder/model.krm", source=series),
                "transformer": Artifact(
                    "transformer/model.krm",
                    source=_synthetic_series(series.parent, self._OWN, "transformer"),
                ),
            },
            weights={
                "text_encoder": {"f16": WeightFiles("text_encoder")},
                "transformer": {"f16": WeightFiles("transformer")},
            },
            assets={},
            quants={
                "f16": {"weights": {"text_encoder": "f16", "transformer": "f16"}, "session": {}}
            },
            default_quant="f16",
            pipeline_config={},
        )

    def _source_dist(self, tmp_path: Path, series: Path) -> Path:
        """The referenced source (the already assembled distribution of another repository — split
        into the same number of parts there)."""
        source = tmp_path / "models" / "karume-source"
        assemble_family([self._plan("source", series)], source, "source")
        return source

    def _components(self, source: Path, **overrides: Any) -> ExternalComponents:
        return ExternalComponents(
            **{
                "repo": self._REPO,
                "revision": self._REVISION,
                "dist": source,
                "model": "source",
                "roles": ("text_encoder",),
                **overrides,
            }
        )

    def test_every_part_becomes_its_own_pinned_reference(self, tmp_path: Path) -> None:
        """repo / revision identical on every element; path / size / sha256 per part; ordered by
        number."""
        series = self._series(tmp_path / "series", self._SHARDS)
        source = self._source_dist(tmp_path, series)
        out_dir = tmp_path / "models" / "karume-borrower"

        manifest = assemble_family(
            [self._plan("borrower", series)],
            out_dir,
            "borrower",
            external=(self._components(source),),
        )

        entry = manifest["models"]["borrower"]["weights"]["text_encoder"]["f16"]
        assert entry["container"]["parts"] == [
            {
                "repo": self._REPO,
                "revision": self._REVISION,
                "path": f"source/text_encoder/model-{index:05d}-of-00003.krm",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for index, payload in enumerate(self._SHARDS, start=1)
        ]

    def test_the_referenced_parts_are_not_stored_here_a_second_time(self, tmp_path: Path) -> None:
        """The very reason references exist — every part stays on the other side, and this
        repository holds only its own role."""
        series = self._series(tmp_path / "series", self._SHARDS)
        source = self._source_dist(tmp_path, series)
        out_dir = tmp_path / "models" / "karume-borrower"

        assemble_family(
            [self._plan("borrower", series)],
            out_dir,
            "borrower",
            external=(self._components(source),),
        )

        assert not (out_dir / "borrower" / "text_encoder").exists()
        assert sorted(verify_dist(out_dir)) == _part_paths(
            "borrower", "transformer/model.krm", self._OWN, "transformer"
        )

    @pytest.mark.parametrize("victim", [0, 1, 2])
    def test_it_checks_every_part_against_the_source_distribution(
        self, tmp_path: Path, victim: int
    ) -> None:
        """The comparison reaches **every element** of the part list (looking at the head only would
        keep tampering in later parts silent)."""
        series = self._series(tmp_path / "series", self._SHARDS)
        source = self._source_dist(tmp_path, series)
        # Rewrites only the referenced source's files, keeping their length (exercises the sha256
        # check, not the size check).
        tampered = source / "source" / "text_encoder" / (f"model-{victim + 1:05d}-of-00003.krm")
        tampered.write_bytes(bytes(len(self._SHARDS[victim])))
        out_dir = tmp_path / "models" / "karume-borrower"

        with pytest.raises(DistError, match="differs from the bytes this assembly builds"):
            assemble_family(
                [self._plan("borrower", series)],
                out_dir,
                "borrower",
                external=(self._components(source),),
            )
        assert not out_dir.exists()

    def test_the_smallest_component_is_referenced_part_by_part(self, tmp_path: Path) -> None:
        """Even a role of a few KB is referenced part by part (always split — there is no single
        reference to the representative path)."""
        whole = tuple(ir_container(mark="whole-text-encoder", named="text_encoder"))
        series = self._series(tmp_path / "series", whole)
        source = self._source_dist(tmp_path, series)
        out_dir = tmp_path / "models" / "karume-borrower"

        manifest = assemble_family(
            [self._plan("borrower", series)],
            out_dir,
            "borrower",
            external=(self._components(source),),
        )

        entry = manifest["models"]["borrower"]["weights"]["text_encoder"]["f16"]
        assert entry["container"]["parts"] == [
            {
                "repo": self._REPO,
                "revision": self._REVISION,
                "path": f"source/text_encoder/model-{index:05d}-of-00003.krm",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for index, payload in enumerate(whole, start=1)
        ]

    def test_an_asset_seat_cannot_point_at_a_split_component(self, tmp_path: Path) -> None:
        """assets is an entry that can only hold a single-file reference — it does not silently
        point at the first part only.

        The form where a component that is split as a weights role in the referenced source is
        pointed at from an assets entry on this side (in a different entry, a role with the same
        spelling has a different meaning).
        """
        series = self._series(tmp_path / "series", self._SHARDS)
        source = self._source_dist(tmp_path, series)
        out_dir = tmp_path / "models" / "karume-borrower"
        # This repository's side is a single file (the split is a fact of the referenced side).
        # The reference is applied from the assets entry.
        base = self._plan("borrower", series)
        borrower = replace(
            base,
            artifacts={
                **base.artifacts,
                "text_encoder": Artifact("text_encoder/model.krm", payload=b"local-copy"),
            },
            weights={"transformer": {"f16": WeightFiles("transformer")}},
            assets={"encoder": "text_encoder"},
            quants={"f16": {"weights": {"transformer": "f16"}, "session": {}}},
        )

        with pytest.raises(
            DistError, match="A cross-repository reference cannot apply to a split-form container"
        ):
            assemble_family([borrower], out_dir, "borrower", external=(self._components(source),))
        assert not out_dir.exists()


class TestTheManifestHubReads:
    """The hub's `karume/5` rules accept the `karume.json` that `karume dist` wrote as is.

    What is checked is the surface **settled by declarations alone** (ADR 0109 decision 6) — the
    limits and the `size: 0` rule are {@link karume.dist.assert_container_limits}, and the form the
    selection looks up (`container.descriptor` / `container.parts`) is spelled here. The TS-side
    parser is not copied (that would make two sources of truth), but **the constants holding the
    same values** are compared mechanically by {@link TestTheHubConstants} below.
    """

    _MARK = "hub-reader"

    def _assembled(self, tmp_path: Path) -> tuple[Path, dict[str, Any]]:
        plan = _synthetic_plan(tmp_path / "series", "A", "dit/model.krm", self._MARK)
        out_dir = tmp_path / "models" / "karume-synthetic"
        return out_dir, assemble_family([plan], out_dir, "A")

    def test_the_written_manifest_passes_the_declared_limits(self, tmp_path: Path) -> None:
        """Reading the placed `karume.json` back passes the same checks (not decoration on the
        return value only)."""
        out_dir, _ = self._assembled(tmp_path)

        written = json.loads((out_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))

        assert written["format"] == "karume/5"
        assert_manifest_limits(written)
        verify_dist(out_dir)

    def test_the_selection_finds_a_descriptor_and_a_part_list(self, tmp_path: Path) -> None:
        """The form `resolveSelection` looks up — quant → component → dtype → `container`."""
        _, manifest = self._assembled(tmp_path)
        model = manifest["models"]["A"]
        label = model["quants"][model["defaultQuant"]]["weights"]["w"]

        container = model["weights"]["w"][label]["container"]

        assert sorted(container) == ["descriptor", "parts"]
        assert sorted(container["descriptor"]) == ["graph", "model"]
        assert len(container["parts"]) >= 2
        assert all(sorted(ref) == ["path", "sha256", "size"] for ref in container["parts"])

    def test_the_descriptor_expectation_matches_the_part_zero_that_was_placed(
        self, tmp_path: Path
    ) -> None:
        """The expectations of the two documents are taken from **the placed files** (table and
        files do not move independently)."""
        out_dir, manifest = self._assembled(tmp_path)
        container = manifest["models"]["A"]["weights"]["w"]["f16"]["container"]

        part0 = out_dir / container["parts"][0]["path"]
        graph, model = read_descriptor_refs(part0)

        assert container["descriptor"]["graph"] == graph.to_document()
        assert container["descriptor"]["model"] == model.to_document()
        # Part 0 is "header + exactly the two documents" (container-v1 §8).
        assert container["parts"][0]["size"] == HEADER_BYTES + graph.length + model.length

    def test_the_parts_open_as_a_container(self, tmp_path: Path) -> None:
        """Passing the declared part list as is opens it as a container (the order MUST be by
        index)."""
        out_dir, manifest = self._assembled(tmp_path)
        container = manifest["models"]["A"]["weights"]["w"]["f16"]["container"]

        parts = [out_dir / ref["path"] for ref in container["parts"]]
        verified = verify.verify_container(parts, blocks=True)

        assert verified.read.model is not None
        assert len(verified.read.model.parts) == len(parts) - 1


def _evaluate_product(expression: str) -> int:
    """Turns an expression of **products and powers only**, such as `1024 * 2 ** 20`, into a number.

    MUST: no `eval` — on the day the TS side changes the spelling of the right-hand side (refers to
    another constant, becomes a function call), the comparison would fail with `SyntaxError` /
    `NameError` instead of "the values differ". What the reader wants to know then is "which value
    moved", not a Python exception name. The accepted set is narrowed to products and powers, so if
    the form of the expression changes, this fails **saying so**.
    """
    total = 1
    tokens = [token.strip() for token in expression.replace("**", "^").split("*")]
    for token in tokens:
        if "^" in token:
            base, power = (part.strip() for part in token.split("^"))
            total *= int(base) ** int(power)
        else:
            total *= int(token)
    return total


class TestTheHubConstants:
    """The container limits have **the same values in the hub and the exporter** ("kept
    independently" in ADR 0109 decision 3).

    The decision spells the values in each language instead of linking them by derivation, so the
    check that fails on the day only one side moves lives here (`packages/hub/src/manifest.ts` is
    text Python can read, so the comparison can be mechanical).
    """

    _MANIFEST_TS: ClassVar[Path] = (
        Path(__file__).resolve().parents[3] / "packages" / "hub" / "src" / "manifest.ts"
    )

    def _declared(self, name: str) -> int:
        source = self._MANIFEST_TS.read_text(encoding="utf-8")
        match = re.search(rf"^const {name} = ([^;]+);", source, re.MULTILINE)
        assert match is not None, f"{name} is not in {self._MANIFEST_TS}"
        return _evaluate_product(match.group(1))

    @pytest.mark.parametrize(
        ("python_value", "typescript_name"),
        [
            (MAX_PARTS, "MAX_PARTS"),
            (PART_MAX_BYTES, "MAX_PART_BYTES"),
            (MAX_DESCRIPTOR_BYTES, "MAX_DESCRIPTOR_BYTES"),
            (HEADER_BYTES, "CONTAINER_HEADER_BYTES"),
        ],
    )
    def test_the_limit_agrees_with_the_hub(self, python_value: int, typescript_name: str) -> None:
        assert python_value == self._declared(typescript_name)

    def test_the_empty_sha256_agrees_with_the_hub(self) -> None:
        """The sha256 a zero-length part declares is uniquely determined (ADR 0109 decision 3)."""
        source = self._MANIFEST_TS.read_text(encoding="utf-8")
        match = re.search(r'^const EMPTY_SHA256 = "([0-9a-f]+)";', source, re.MULTILINE)
        assert match is not None

        assert match.group(1) == dist.EMPTY_SHA256


class TestTheHubConstantReader:
    """The reader of the constant comparison itself (an accepted set without `eval`).

    If the day the TS side changes the spelling of the right-hand side failed with a Python
    exception name instead of "the values differ", what the reader wants to know (which value
    moved) would vanish from the diagnostic.
    """

    @pytest.mark.parametrize(
        ("expression", "value"),
        [("1024", 1024), ("1024 * 1024", 1048576), ("1024 * 2 ** 20", 1073741824), ("24", 24)],
    )
    def test_a_product_of_literals_is_evaluated(self, expression: str, value: int) -> None:
        assert _evaluate_product(expression) == value

    @pytest.mark.parametrize("expression", ["MAX_PARTS", "1 + 1", "readLimit()"])
    def test_anything_but_a_product_of_literals_fails_loudly(self, expression: str) -> None:
        with pytest.raises(ValueError):
            _evaluate_product(expression)


class TestTheRepositorySpelling:
    """Each segment of `<owner>/<name>` uses the same allow list as the hub's `REPO_SEGMENT_RE`.

    The hub rejects segments with **a leading dot** (including `.` / `..` themselves) — the `/` of
    a repo is treated as a structural element by the fetch layer and is not percent-encoded, so
    letting it through would mean traversal outside the SHA pin. If the Python side let it
    through, it could build a distribution whose assembly is green but whose manifest the hub
    cannot open.
    """

    @pytest.mark.parametrize("repo", [".hidden/name", "owner/.name", "../name", "owner/.."])
    def test_a_segment_with_a_leading_dot_is_refused(self, repo: str) -> None:
        with pytest.raises(DistError, match="<owner>/<name>"):
            resolve_repo(_PER_MODEL_REPO_PIPELINE, ["m"], repo)

    @pytest.mark.parametrize("repo", ["hdae/karume.voices", "hdae/karume-voices_2", "_o/n"])
    def test_a_dot_after_the_first_character_is_accepted(self, repo: str) -> None:
        """Counterpart — a dot anywhere but first is accepted by the hub too (no over-rejection)."""
        assert resolve_repo(_PER_MODEL_REPO_PIPELINE, ["m"], repo) == repo


class TestATrailingNewlineIsRefused:
    """The spelling checks reject a trailing newline — aligning the accepted set with the hub (JS
    `$`).

    Python's `$` also matches right before a trailing newline, so looking up `^…$` with `.match`
    lets `'owner/name\\n'` through. The hub rejects it, so assembly would be green while the hub
    cannot open the manifest.
    """

    def test_a_repository_with_a_trailing_newline_is_refused(self) -> None:
        with pytest.raises(DistError, match="<owner>/<name>"):
            resolve_repo(_PER_MODEL_REPO_PIPELINE, ["m"], "owner/name\n")

    def test_a_model_name_with_a_trailing_newline_is_refused(self) -> None:
        with pytest.raises(DistError, match="allowed characters"):
            assert_model_name("name\n")

    def test_a_relative_path_with_a_trailing_newline_is_refused(self) -> None:
        with pytest.raises(DistError, match="allowed characters"):
            dist.assert_rel_path("a/b.json\n", "test")

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("repo", "owner/name\n", "<owner>/<name>"),
            ("revision", "0" * 40 + "\n", "40-digit commit sha"),
        ],
    )
    def test_an_external_reference_with_a_trailing_newline_is_refused(
        self, field: str, value: str, message: str
    ) -> None:
        ref = {
            "repo": "owner/name",
            "revision": "0" * 40,
            "path": "a/b.json",
            "size": 1,
            "sha256": "0" * 64,
        }

        with pytest.raises(DistError, match=message):
            dist.assert_external_ref("test", {**ref, field: value})

    def test_the_same_reference_without_the_newline_is_accepted(self) -> None:
        """Counterpart — the same spelling without the newline passes (confirms the newline is the
        only reason for the rejection)."""
        dist.assert_external_ref(
            "test",
            {
                "repo": "owner/name",
                "revision": "0" * 40,
                "path": "a/b.json",
                "size": 1,
                "sha256": "0" * 64,
            },
        )
        assert resolve_repo(_PER_MODEL_REPO_PIPELINE, ["m"], "owner/name") == "owner/name"
