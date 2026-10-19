"""The whole-repository mode of the migration CLI (`karume migrate --manifest`) — ADR 0109
decisions 3 / 4 / 8.

The subject is **a synthetic `karume/4` repository** (the real mirror is not read). Converting a
single graph is the job of `test_migrate.py`, so what this file checks are the facts that appear
only at repository level:

- the shard sequences are built from the manifest's declarations (sequences spanning directories,
  several models pointing at the same sequence folded into one)
- `extras` and the PLE sidecar become container assets and disappear from the manifest's `assets`
- cross-repository references are not converted but copied from the converted directory given by
  `--cross-repo`
- files that were not converted are copied as is, and a `karume/5` karume.json is written
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from legacy_writer import Entry, legacy_fill_shards, legacy_shards, order, write_safetensors

from karume import migrate, publish
from karume.container import PART_LENGTH_CHOICES, AssetRecord, Provenance, numbered_name
from karume.legacy import LegacyFormatError
from karume.migrate import MigrateError, migrate_repository, parse_cross_repo
from karume.verify import ContainerError

PROVENANCE = Provenance(license="apache-2.0", writer="karume/test")

#: Sizes lowered so that the synthetic assets (a few KiB) exercise part crossings and the block
#: split of assets.
SMALL_PART_BYTES = 1024
SMALL_BLOCK_BYTES = 512

MIB = 1024 * 1024

#: The pin of a cross-repository reference (40 lowercase hex digits — the form `dist.REVISION_RE`
#: accepts).
REVISION = "0" * 39 + "1"

#: The synthetic PLE sizes (one `values` row = layers × dim / 2 = 16 B, one `scales` row =
#: layers × 4 = 8 B).
PLE_TOKENS = 64
PLE_LAYERS = 2
PLE_DIM = 16
PLE_RANGES: tuple[tuple[int, int], ...] = ((0, 20), (20, 45), (45, 64))


# ---------------------------------------------------------------------------
# The synthetic `karume/4` repository
# ---------------------------------------------------------------------------


def place(root: Path, rel: str, blob: bytes) -> dict[str, Any]:
    """Place one file and return its triple (path / size / sha256)."""
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)
    return {"path": rel, "size": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def stage_component(
    root: Path, directories: Sequence[str], stem: str, shards: Sequence[bytes]
) -> list[dict[str, Any]]:
    """Lay the shard sequence out **in each shard's own directory** and return the FileRefs (a
    sequence can span directories)."""
    total = len(shards)
    return [
        place(root, numbered_name(f"{directory}/{stem}.safetensors", index, total), blob)
        for index, (directory, blob) in enumerate(zip(directories, shards, strict=True), start=1)
    ]


def quant(weights: Mapping[str, str]) -> dict[str, Any]:
    return {"weights": dict(weights), "session": {}}


def model_entry(
    weights: Mapping[str, Mapping[str, Any]],
    assets: Mapping[str, Any],
    quants: Mapping[str, Any],
    *,
    pipeline: str = "synthetic",
) -> dict[str, Any]:
    return {
        "pipeline": f"{pipeline}/1",
        "weights": {name: dict(labels) for name, labels in weights.items()},
        "assets": dict(assets),
        "quants": dict(quants),
        "defaultQuant": next(iter(quants)),
        "pipelineConfig": {"kind": "synthetic"},
    }


def write_manifest(root: Path, models: Mapping[str, Any], default_model: str) -> Path:
    manifest = {
        "format": "karume/4",
        "generator": "karume/0.0.0",
        "defaultModel": default_model,
        "models": dict(models),
    }
    path = root / "karume.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def small_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Add **the synthetic assets' size** to the writer's choice set (256 MiB and up) (every case
    in this module).

    A synthetic repository of a few KiB does not split into parts at the set's part lengths, so
    the set is widened to let `migrated`'s size through. The check of the set itself (values
    outside it fail) is done by `TestThePartLength` with the real set (that class cancels this
    fixture with a fixture of the same name).
    """
    monkeypatch.setattr(migrate, "PART_LENGTH_CHOICES", (*PART_LENGTH_CHOICES, SMALL_PART_BYTES))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository in which 2 models share `encoder` and the `decoder` sequence spans
    directories."""
    root = tmp_path / "src"
    encoder = stage_component(
        root, ["shared/encoder"] * 2, "model.f32", legacy_shards(mark="enc", storage="f32")
    )
    # A sequence spanning directories (the graph shard in the shared directory, the weight shard
    # in the speaker directory).
    decoder = stage_component(
        root, ["shared/decoder", "alpha/decoder"], "model.f32", legacy_shards(mark="dec")
    )
    rope = place(root, "alpha/decoder/rope_base.safetensors", b"rope-base-table\n")
    tokenizer = place(root, "tokenizer.json", b'{"model":"synthetic"}\n')
    (root / "README.md").write_text("# synthetic\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")

    alpha = model_entry(
        {
            "encoder": {"f32": {"shards": encoder}},
            "decoder": {"f32": {"shards": decoder, "extras": {"rope_base": rope}}},
        },
        {"tokenizer": tokenizer},
        {"full": quant({"encoder": "f32", "decoder": "f32"})},
    )
    beta = model_entry(
        {"encoder": {"f32": {"shards": encoder}}},
        {"tokenizer": tokenizer},
        {"full": quant({"encoder": "f32"})},
    )
    write_manifest(root, {"alpha": alpha, "beta": beta}, "alpha")
    return root


def migrated(repo: Path, out: Path, **overrides: Any) -> migrate.RepositoryResult:
    return migrate_repository(
        repo / "karume.json",
        out,
        provenance=PROVENANCE,
        part_bytes=SMALL_PART_BYTES,
        _block_bytes=SMALL_BLOCK_BYTES,
        **overrides,
    )


def manifest_of(out: Path) -> dict[str, Any]:
    return json.loads((out / "karume.json").read_text(encoding="utf-8"))


def container_of(out: Path, model: str, component: str, dtype: str) -> dict[str, Any]:
    entry = manifest_of(out)["models"][model]["weights"][component][dtype]
    assert set(entry) == {"container"}
    return entry["container"]


def opened(out: Path, container: Mapping[str, Any]) -> Any:
    from karume.container import read_container

    return read_container([out / ref["path"] for ref in container["parts"]])


def asset_payload(read: Any, name: str) -> bytes:
    """The payload of one asset (cut at the declared logical length — trailing 0x00 bytes are not
    stripped by guesswork)."""
    record = read.model.assets[name]
    return read.block(record.block)[: record.length]


# ---------------------------------------------------------------------------


class TestTheRepositoryManifest:
    def test_it_writes_karume_5_with_the_old_declarations_kept(
        self, repo: Path, tmp_path: Path
    ) -> None:
        result = migrated(repo, tmp_path / "out")
        manifest = manifest_of(tmp_path / "out")

        assert result.manifest == tmp_path / "out" / "karume.json"
        assert manifest["format"] == "karume/5"
        assert manifest["defaultModel"] == "alpha"
        assert list(manifest["models"]) == ["alpha", "beta"]
        alpha = manifest["models"]["alpha"]
        assert list(alpha) == [
            "pipeline",
            "weights",
            "assets",
            "quants",
            "defaultQuant",
            "pipelineConfig",
        ]
        assert alpha["quants"] == {"full": quant({"encoder": "f32", "decoder": "f32"})}
        assert alpha["pipelineConfig"] == {"kind": "synthetic"}
        assert set(alpha["assets"]) == {"tokenizer"}

    def test_the_container_entry_has_the_declared_shape(self, repo: Path, tmp_path: Path) -> None:
        """ADR 0109 decision 3 — the expected values of each of the 2 documents + the FileRefs of
        every part including part 0."""
        out = tmp_path / "out"
        migrated(repo, out)
        container = container_of(out, "alpha", "encoder", "f32")

        assert list(container) == ["descriptor", "parts"]
        assert list(container["descriptor"]) == ["graph", "model"]
        for document in container["descriptor"].values():
            assert list(document) == ["length", "sha256"]
        read = opened(out, container)
        assert container["descriptor"]["graph"] == {
            "length": len(read.graph_descriptor_bytes),
            "sha256": hashlib.sha256(read.graph_descriptor_bytes).hexdigest(),
        }
        assert container["descriptor"]["model"] == {
            "length": len(read.model_descriptor_bytes),
            "sha256": hashlib.sha256(read.model_descriptor_bytes).hexdigest(),
        }
        assert len(container["parts"]) == len(read.model.parts) + 1
        for ref in container["parts"]:
            blob = (out / ref["path"]).read_bytes()
            assert ref["size"] == len(blob)
            assert ref["sha256"] == hashlib.sha256(blob).hexdigest()

    def test_a_zero_length_part_is_written_and_declared_with_size_zero(
        self, tmp_path: Path
    ) -> None:
        """Even with an empty const region, part 1 is listed as a 0-byte file (ADR 0109 decision 3).

        The subject is a component **with no constant at all** (`legacy_fill_shards`) — a
        synthesis with consts would fill part 1 and could not exercise the 0-byte part itself.
        """
        root = tmp_path / "empty-const"
        plain = stage_component(
            root, ["alpha/encoder"] * 2, "model.f32", legacy_fill_shards(2, mark="plain")
        )
        write_manifest(
            root,
            {
                "alpha": model_entry(
                    {"encoder": {"f32": {"shards": plain}}},
                    {},
                    {"full": quant({"encoder": "f32"})},
                )
            },
            "alpha",
        )
        out = tmp_path / "out"
        migrated(root, out)
        parts = container_of(out, "alpha", "encoder", "f32")["parts"]

        assert parts[1]["size"] == 0
        assert (out / parts[1]["path"]).stat().st_size == 0
        assert parts[1]["sha256"] == hashlib.sha256(b"").hexdigest()

    def test_the_output_sits_beside_the_old_weight_shards(self, repo: Path, tmp_path: Path) -> None:
        """The output goes to **the parent of the weight shards** (no guessing even when the
        sequence spans directories)."""
        out = tmp_path / "out"
        migrated(repo, out)

        assert container_of(out, "alpha", "encoder", "f32")["parts"][0]["path"].startswith(
            "shared/encoder/model.f32-"
        )
        # decoder has its graph shard in shared/ and its weight shard in alpha/.
        assert container_of(out, "alpha", "decoder", "f32")["parts"][0]["path"].startswith(
            "alpha/decoder/model.f32-"
        )

    def test_a_shared_component_is_converted_once(self, repo: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        result = migrated(repo, out)

        # Only encoder (shared) + decoder, 2 in all, are converted (there are 3 entries).
        assert len(result.converted) == 2
        assert container_of(out, "alpha", "encoder", "f32") == container_of(
            out, "beta", "encoder", "f32"
        )

    def test_the_same_input_produces_the_same_bytes(self, repo: Path, tmp_path: Path) -> None:
        first, second = tmp_path / "a", tmp_path / "b"
        migrated(repo, first)
        migrated(repo, second)

        def tree(root: Path) -> dict[str, bytes]:
            return {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in sorted(root.rglob("*"))
                if path.is_file()
            }

        assert tree(first) == tree(second)

    def test_the_old_repository_is_read_only(self, repo: Path, tmp_path: Path) -> None:
        before = {
            path.relative_to(repo).as_posix(): path.read_bytes()
            for path in sorted(repo.rglob("*"))
            if path.is_file()
        }
        migrated(repo, tmp_path / "out")

        assert {
            path.relative_to(repo).as_posix(): path.read_bytes()
            for path in sorted(repo.rglob("*"))
            if path.is_file()
        } == before

    def test_a_manifest_that_is_not_karume_4_fails_loudly(self, tmp_path: Path) -> None:
        root = tmp_path / "src"
        root.mkdir()
        (root / "karume.json").write_text('{"format":"karume/5"}\n', encoding="utf-8")

        with pytest.raises(MigrateError, match="only 'karume/4' can be migrated"):
            migrated(root, tmp_path / "out")

    def test_an_output_inside_the_input_fails_loudly(self, repo: Path) -> None:
        with pytest.raises(MigrateError, match="is nested with"):
            migrated(repo, repo / "out")

    def test_weight_shards_spread_over_directories_fail_loudly(self, tmp_path: Path) -> None:
        """The output goes to the one parent of the weight shards (if they are scattered, it stops
        instead of guessing)."""
        root = tmp_path / "src"
        spread = stage_component(
            root,
            ["shared/multi", "alpha/multi", "beta/multi"],
            "model.f32",
            legacy_fill_shards(3, mark="multi"),
        )
        write_manifest(
            root,
            {
                "alpha": model_entry(
                    {"multi": {"f32": {"shards": spread}}}, {}, {"full": quant({"multi": "f32"})}
                )
            },
            "alpha",
        )

        with pytest.raises(
            MigrateError, match="the parent directories of the weight shards differ"
        ):
            migrated(root, tmp_path / "out")

    def test_a_shard_row_with_mixed_stems_fails_loudly(self, tmp_path: Path) -> None:
        """The output stem is taken from the old shard names (if they differ, it stops instead of
        guessing)."""
        root = tmp_path / "src"
        blobs = legacy_shards(mark="mix")
        mixed = [
            place(root, numbered_name("shared/mix/model.f32.safetensors", 1, 2), blobs[0]),
            place(root, numbered_name("shared/mix/other.f32.safetensors", 2, 2), blobs[1]),
        ]
        write_manifest(
            root,
            {
                "alpha": model_entry(
                    {"mix": {"f32": {"shards": mixed}}}, {}, {"full": quant({"mix": "f32"})}
                )
            },
            "alpha",
        )

        with pytest.raises(MigrateError, match="stems of the shard sequence differ"):
            migrated(root, tmp_path / "out")

    def test_the_single_form_is_refused(self, repo: Path, tmp_path: Path) -> None:
        """container.parts of karume/5 MUST have 2 or more elements (ADR 0109 decision 3)."""
        with pytest.raises(MigrateError, match="--single cannot be used in whole-repository mode"):
            migrate.main(
                [
                    "--manifest",
                    str(repo / "karume.json"),
                    "--out",
                    str(tmp_path / "out"),
                    "--license",
                    "apache-2.0",
                    "--single",
                ]
            )

    def test_a_failure_midway_through_the_swap_leaves_nothing_behind(
        self, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed run leaves nothing behind (no container half-published under its final name —
        §12)."""
        out = tmp_path / "out"
        real = os.replace
        seen = 0

        def flaky(source: Any, target: Any) -> None:
            nonlocal seen
            seen += 1
            if seen == 2:
                raise OSError("a run that failed midway through the swap")
            real(source, target)

        monkeypatch.setattr(publish.os, "replace", flaky)
        with pytest.raises(OSError, match="midway through the swap"):
            migrated(repo, out)

        assert not list(out.rglob("*.krm"))


class TestTheCopiedFiles:
    def test_plain_files_are_copied_and_converted_ones_are_not(
        self, repo: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        result = migrated(repo, out)
        copied = {path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()}

        assert "README.md" in copied
        assert "tokenizer.json" in copied
        # Directories starting with `.` are not included.
        assert not any(name.startswith(".git/") for name in copied)
        # The converted shards, the extras that became container assets and the old karume.json
        # are not copied (the new karume.json is written).
        assert not any(name.endswith(".safetensors") for name in copied)
        assert not (out / "alpha" / "decoder" / "rope_base.safetensors").exists()
        assert result.copied == 2  # README.md / tokenizer.json
        assert (out / "karume.json").is_file()

    def test_it_refuses_to_write_over_a_previous_output(self, repo: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        migrated(repo, out)

        with pytest.raises(
            MigrateError,
            match=r"already exists in the output directory|Artifacts of a previous run",
        ):
            migrated(repo, out)


class TestTheDeclaredFiles:
    """The old manifest's FileRefs (size / sha256) are matched against the actual files before
    converting.

    The migration's "raw bytes identical" is a comparison with the local files, so without this a
    mirror whose contents disagree with the old revision would be laundered into a new sha256, and
    the FileRefs of the assets copied verbatim would claim a sha256 different from the actual
    files. The match runs before the conversion — when it fails, out holds no container at all.
    """

    def test_a_flipped_byte_in_an_old_shard_fails_before_any_container_is_written(
        self, repo: Path, tmp_path: Path
    ) -> None:
        # The weight shard of a later entry in declaration order (alpha.decoder) — checks that it
        # fails before the first entry is converted.
        shard = repo / numbered_name("alpha/decoder/model.f32.safetensors", 2, 2)
        blob = bytearray(shard.read_bytes())
        blob[-1] ^= 0xFF
        shard.write_bytes(bytes(blob))
        out = tmp_path / "out"

        with pytest.raises(
            MigrateError,
            match=(
                r"'alpha/decoder/model\.f32-00002-of-00002\.safetensors'"
                r" declared by the old manifest has a sha256"
            ),
        ):
            migrated(repo, out)
        assert not list(out.rglob("*.krm"))

    def test_a_copied_file_whose_size_differs_from_its_declaration_fails_loudly(
        self, repo: Path, tmp_path: Path
    ) -> None:
        (repo / "tokenizer.json").write_text('{"model":"changed"}\n', encoding="utf-8")
        out = tmp_path / "out"

        with pytest.raises(
            MigrateError,
            match=(
                r"'tokenizer\.json' declared by the old manifest has length \d+,"
                r" not the declared \d+"
            ),
        ):
            migrated(repo, out)
        assert not list(out.rglob("*.krm"))

    @pytest.mark.parametrize("spelling", ["a" * 63, "A" * 64])
    def test_a_sha256_that_is_not_64_lowercase_hex_digits_fails_loudly(
        self, repo: Path, tmp_path: Path, spelling: str
    ) -> None:
        manifest = json.loads((repo / "karume.json").read_text(encoding="utf-8"))
        manifest["models"]["alpha"]["assets"]["tokenizer"]["sha256"] = spelling
        (repo / "karume.json").write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(MigrateError, match=r"assets\['tokenizer'\]\.sha256 is not a sha256"):
            migrated(repo, tmp_path / "out")

    def test_a_crossed_descriptor_sha256_is_spelled_like_a_sha256(
        self, borrower: Path, repo: Path, tmp_path: Path
    ) -> None:
        """The spelling of the sha256 on the copied side (the referenced `karume/5`) is checked too
        — it is a value copied verbatim."""
        lender_out = tmp_path / "lender"
        migrated(repo, lender_out)
        manifest = json.loads((lender_out / "karume.json").read_text(encoding="utf-8"))
        container = manifest["models"]["alpha"]["weights"]["encoder"]["f32"]["container"]
        container["descriptor"]["model"]["sha256"] = "a" * 63
        (lender_out / "karume.json").write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(MigrateError, match=r"descriptor\.model\.sha256 is not a sha256"):
            migrated(
                borrower,
                tmp_path / "out",
                cross_repos=[parse_cross_repo(f"hdae/lender={lender_out}@{REVISION}")],
            )


class TestTheExtras:
    def test_rope_base_becomes_a_container_asset(self, repo: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        migrated(repo, out)
        read = opened(out, container_of(out, "alpha", "decoder", "f32"))

        expected = (repo / "alpha/decoder/rope_base.safetensors").read_bytes()
        assert read.model.assets == {
            "rope_base": AssetRecord(block="a.0", role="rope-base", length=len(expected))
        }
        raw = read.block("a.0")
        assert raw[: len(expected)] == expected
        assert raw[len(expected) :] == b"\x00" * (len(raw) - len(expected))
        # extras disappear from the manifest entry (the container holds them).
        assert "extras" not in container_of(out, "alpha", "decoder", "f32")

    def test_an_unknown_extra_fails_loudly(self, repo: Path, tmp_path: Path) -> None:
        manifest = json.loads((repo / "karume.json").read_text(encoding="utf-8"))
        entry = manifest["models"]["alpha"]["weights"]["decoder"]["f32"]
        entry["extras"] = {"mystery": entry["extras"]["rope_base"]}
        (repo / "karume.json").write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(MigrateError, match="extras 'mystery' has no mapping target"):
            migrated(repo, repo.parent / "out")


# ---------------------------------------------------------------------------
# Cross-repository references
# ---------------------------------------------------------------------------


@pytest.fixture
def borrower(tmp_path: Path, repo: Path) -> Path:
    """A repository that borrows `alpha.encoder.f32` from `hdae/lender` (one weight of its own)."""
    root = tmp_path / "borrower"
    own = stage_component(root, ["gamma/decoder"] * 2, "model.f32", legacy_shards(mark="own"))
    lender = json.loads((repo / "karume.json").read_text(encoding="utf-8"))
    borrowed = [
        {**ref, "repo": "hdae/lender", "revision": "a" * 40}
        for ref in lender["models"]["alpha"]["weights"]["encoder"]["f32"]["shards"]
    ]
    alpha = model_entry(
        {
            "encoder": {"f32": {"shards": borrowed}},
            "decoder": {"f32": {"shards": own}},
        },
        {},
        {"full": quant({"encoder": "f32", "decoder": "f32"})},
    )
    write_manifest(root, {"alpha": alpha}, "alpha")
    return root


class TestTheCrossRepoReferences:
    def test_an_undeclared_repo_is_listed_with_its_count(
        self, borrower: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(MigrateError, match=r"hdae/lender \(1 slots\)"):
            migrated(borrower, tmp_path / "out")

    def test_the_container_is_copied_with_the_new_pin(
        self, borrower: Path, repo: Path, tmp_path: Path
    ) -> None:
        lender_out = tmp_path / "lender"
        migrated(repo, lender_out)
        out = tmp_path / "out"
        migrated(
            borrower,
            out,
            cross_repos=[parse_cross_repo(f"hdae/lender={lender_out}@{REVISION}")],
        )
        crossed = container_of(out, "alpha", "encoder", "f32")

        assert (
            crossed["descriptor"]
            == container_of(lender_out, "alpha", "encoder", "f32")["descriptor"]
        )
        for ref, source in zip(
            crossed["parts"],
            container_of(lender_out, "alpha", "encoder", "f32")["parts"],
            strict=True,
        ):
            assert ref == {**source, "repo": "hdae/lender", "revision": REVISION}
        # A cross-repository sequence is not placed in the own repository.
        assert not (out / crossed["parts"][0]["path"]).exists()

    def test_the_lookup_follows_the_output_path_not_the_seat_name(
        self, borrower: Path, repo: Path, tmp_path: Path
    ) -> None:
        """The key for copying is part 0's location (nothing guarantees that the lender and the
        borrower use the same entry names)."""
        lender_out = tmp_path / "lender"
        migrated(repo, lender_out)
        manifest = json.loads((lender_out / "karume.json").read_text(encoding="utf-8"))
        # Move all of the lender's entries to a model with another name (the container paths do
        # not change).
        manifest["models"] = {"renamed": manifest["models"]["alpha"]}
        manifest["defaultModel"] = "renamed"
        (lender_out / "karume.json").write_text(json.dumps(manifest), encoding="utf-8")
        out = tmp_path / "out"
        migrated(
            borrower,
            out,
            cross_repos=[parse_cross_repo(f"hdae/lender={lender_out}@{REVISION}")],
        )
        crossed = container_of(out, "alpha", "encoder", "f32")

        assert crossed["parts"][0]["path"].startswith("shared/encoder/model.f32-")
        assert all(ref["repo"] == "hdae/lender" for ref in crossed["parts"])

    def test_a_container_that_is_not_in_the_converted_directory_lists_the_candidates(
        self, borrower: Path, repo: Path, tmp_path: Path
    ) -> None:
        lender_out = tmp_path / "lender"
        migrated(repo, lender_out)
        manifest = json.loads((lender_out / "karume.json").read_text(encoding="utf-8"))
        # The sequence the borrower points at (shared/encoder) is gone from the referenced
        # repository.
        for model in manifest["models"].values():
            model["weights"].pop("encoder", None)
        (lender_out / "karume.json").write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(
            MigrateError,
            match=(
                r"container to copy, .shared/encoder/model\.f32\.krm., does not exist"
                r".*alpha/decoder/model\.f32\.krm"
            ),
        ):
            migrated(
                borrower,
                tmp_path / "out",
                cross_repos=[parse_cross_repo(f"hdae/lender={lender_out}@{REVISION}")],
            )

    def test_a_mixed_entry_fails_loudly(self, borrower: Path, tmp_path: Path) -> None:
        manifest = json.loads((borrower / "karume.json").read_text(encoding="utf-8"))
        shards = manifest["models"]["alpha"]["weights"]["encoder"]["f32"]["shards"]
        shards[1] = {key: shards[1][key] for key in ("path", "size", "sha256")}
        (borrower / "karume.json").write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(MigrateError, match="cross-repository references are mixed"):
            migrated(borrower, tmp_path / "out")

    def test_the_same_repo_twice_fails_loudly(self, borrower: Path, tmp_path: Path) -> None:
        """It cannot be decided which converted directory to copy from."""
        spec = parse_cross_repo(f"hdae/lender={tmp_path / 'lender'}@{REVISION}")

        with pytest.raises(MigrateError, match="the same repo twice"):
            migrated(borrower, tmp_path / "out", cross_repos=[spec, spec])

    @pytest.mark.parametrize(
        "spec",
        [
            "hdae/lender",
            "hdae/lender=/tmp/x",
            "lender=/tmp/x@" + "a" * 40,
            "hdae/lender=/tmp/x@abc",
        ],
    )
    def test_a_malformed_cross_repo_option_fails_loudly(self, spec: str) -> None:
        with pytest.raises(MigrateError, match="--cross-repo"):
            parse_cross_repo(spec)


# ---------------------------------------------------------------------------
# PLE sidecar → container assets (index schema 3)
# ---------------------------------------------------------------------------


def ple_rows(start: int, stop: int, width: int, seed: int) -> bytes:
    """The token rows themselves (determined by `start`, so the old / new match works row by
    row)."""
    return bytes(
        ((token * 31 + offset * 7 + seed) & 0xFF)
        for token in range(start, stop)
        for offset in range(width)
    )


def ple_common(layers: int = PLE_LAYERS, dim: int = PLE_DIM, *, schema: int = 2) -> dict[str, Any]:
    """The fields common to the old index and the shard metadata. Schema 1 (I8) has no `storage`
    field (the shape of the real assets)."""
    return {
        "schema": schema,
        **({"storage": "i4"} if schema == 2 else {}),
        "tokens": PLE_TOKENS,
        "layers": layers,
        "dim": dim,
        "embedScale": 4.0,
    }


def stage_ple(
    root: Path,
    directory: str,
    *,
    layers: int = PLE_LAYERS,
    dim: int = PLE_DIM,
    schema: int = 2,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Place an old PLE sidecar (index + token-range shards) and return `(the index FileRef,
    assets)`.

    `layers` / `dim` can be varied in order to exercise the check decided by the row byte count
    (MUST be a multiple of 4).
    """
    values_row = layers * dim // (2 if schema == 2 else 1)
    values_dtype = "I4" if schema == 2 else "I8"
    scales_row = layers * 4
    assets: dict[str, dict[str, Any]] = {}
    shards: list[dict[str, Any]] = []
    for position, (start, stop) in enumerate(PLE_RANGES, start=1):
        rows = stop - start
        name = numbered_name("ple.safetensors", position, len(PLE_RANGES))
        path = root / directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payloads = {
            "values": ple_rows(start, stop, values_row, 11),
            "scales": ple_rows(start, stop, scales_row, 23),
        }
        blob = write_safetensors(
            order(
                [
                    Entry("values", values_dtype, (rows, layers, dim), payloads["values"]),
                    Entry("scales", "F32", (rows, layers), payloads["scales"]),
                ]
            ),
            {
                "karume_ple": json.dumps(
                    {**ple_common(layers, dim, schema=schema), "start": start, "stop": stop}
                )
            },
        )
        path.write_bytes(blob)
        assets[name] = {
            "path": f"{directory}/{name}",
            "size": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
        }
        shards.append({"file": name, "start": start, "stop": stop})
    index = place(
        root,
        f"{directory}/ple.json",
        (
            json.dumps({**ple_common(layers, dim, schema=schema), "shards": shards}, indent=2)
            + "\n"
        ).encode("utf-8"),
    )
    return index, assets


@pytest.fixture
def gemma_repo(tmp_path: Path) -> Path:
    return stage_gemma_repo(tmp_path / "gem")


def stage_gemma_repo(
    root: Path, *, layers: int = PLE_LAYERS, dim: int = PLE_DIM, schema: int = 2
) -> Path:
    """A repository with a `gemma4` PLE (component `model` with 2 entries, f32 / f16)."""
    weights = {
        dtype: {
            "shards": stage_component(
                root, ["e2b/model"] * 2, f"model.{dtype}", legacy_shards(mark="g", storage=dtype)
            )
        }
        for dtype in ("f32", "f16")
    }
    index, ple = stage_ple(root, "e2b/ple", layers=layers, dim=dim, schema=schema)
    tokenizer = place(root, "tokenizer.json", b"{}\n")
    entry = model_entry(
        {"model": weights},
        {"tokenizer": tokenizer, "ple_index": index, **ple},
        {"full": quant({"model": "f32"}), "half": quant({"model": "f16"})},
        pipeline="gemma4",
    )
    write_manifest(root, {"e2b": entry}, "e2b")
    return root


def redeclare_asset(repo: Path, model: str, asset: str, blob: bytes) -> None:
    """Rewrite a declared asset and update the manifest's triple to match.

    A mismatch between the declaration and the actual file is caught first by the match at the
    migration's entry, so a fault injection that exercises the content checks rewrites the
    manifest too.
    """
    manifest_path = repo / "karume.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = manifest["models"][model]["assets"]
    assets[asset] = place(repo, assets[asset]["path"], blob)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def doctor_ple_shard(repo: Path, position: int, edit: Any) -> None:
    """Rewrite only the header JSON of the old PLE shard `position` with `edit`, and update the
    manifest to match.

    The data section does not move by a single byte (the header is padded with spaces to a
    multiple of 8, as originally), so only the rewritten declaration is broken.
    """
    name = numbered_name("ple.safetensors", position, len(PLE_RANGES))
    blob = (repo / "e2b" / "ple" / name).read_bytes()
    length = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8 : 8 + length])
    edit(header)
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    redeclare_asset(repo, "e2b", name, struct.pack("<Q", len(raw)) + raw + blob[8 + length :])


def source_rows(repo: Path, directory: str, key: str) -> bytes:
    """The raw bytes of `key` in the old shards concatenated in token order (the counterpart of the
    match, taken by another path)."""
    out = bytearray()
    for position in range(1, len(PLE_RANGES) + 1):
        blob = (
            repo / directory / numbered_name("ple.safetensors", position, len(PLE_RANGES))
        ).read_bytes()
        length = struct.unpack("<Q", blob[:8])[0]
        spec = json.loads(blob[8 : 8 + length])[key]
        begin, end = spec["data_offsets"]
        out += blob[8 + length + begin : 8 + length + end]
    return bytes(out)


class TestThePleFold:
    def test_the_index_becomes_schema_3_over_asset_blocks(
        self, gemma_repo: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        migrated(gemma_repo, out)
        read = opened(out, container_of(out, "e2b", "model", "f32"))
        index = json.loads(asset_payload(read, "ple_index"))

        assert list(index) == [
            "schema",
            "storage",
            "tokens",
            "layers",
            "dim",
            "embedScale",
            "values",
            "scales",
        ]
        assert index["schema"] == 3
        assert index["storage"] == "i4"
        assert index["tokens"] == PLE_TOKENS
        assert index["values"]["rowBytes"] == PLE_LAYERS * PLE_DIM // 2
        assert index["scales"]["rowBytes"] == PLE_LAYERS * 4
        # The old shard boundaries (20 / 45) are gone; only the block limit (512 B) decides the
        # step.
        assert [(b["start"], b["stop"]) for b in index["values"]["blocks"]] == [(0, 32), (32, 64)]
        assert [(b["start"], b["stop"]) for b in index["scales"]["blocks"]] == [(0, 64)]
        for key in ("values", "scales"):
            for block in index[key]["blocks"]:
                assert read.model.assets[block["asset"]].role == f"ple-{key}"

    def test_every_token_row_survives_byte_for_byte(self, gemma_repo: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        migrated(gemma_repo, out)
        read = opened(out, container_of(out, "e2b", "model", "f32"))
        index = json.loads(asset_payload(read, "ple_index"))

        for key in ("values", "scales"):
            joined = b"".join(asset_payload(read, block["asset"]) for block in index[key]["blocks"])
            assert joined == source_rows(gemma_repo, "e2b/ple", key)

    def test_a_range_read_block_sits_alone_in_its_part(
        self, gemma_repo: Path, tmp_path: Path
    ) -> None:
        """A block that needs range reads gets one part to itself (ADR 0109 decision 4 /
        container-v1 §4.2).

        A scan-type source (the default on Deno and Node) cuts a part from its start, so sharing
        the part would mean skipping from the part's start to fetch one row — the very failure
        that §4.2 placed this MUST against.
        """
        out = tmp_path / "out"
        migrated(gemma_repo, out)
        read = opened(out, container_of(out, "e2b", "model", "f32"))
        index = json.loads(asset_payload(read, "ple_index"))
        occupants = Counter(block.part for block in read.model.blocks)
        part_of = {block.id: block.part for block in read.model.blocks}

        ranged = [
            read.model.assets[block["asset"]].block
            for key in ("values", "scales")
            for block in index[key]["blocks"]
        ]
        # 2 values + 1 scales (pinned so that the check cannot pass vacuously).
        assert len(ranged) == 3
        for block_id in ranged:
            assert occupants[part_of[block_id]] == 1

        # The index, read whole, does not get a dedicated part, but it does not share one with
        # weight blocks.
        index_part = part_of[read.model.assets["ple_index"].block]
        assert index_part not in {
            part_of[block.id] for block in read.model.blocks if block.role != "asset"
        }

    def test_a_schema_1_i8_sidecar_folds_with_storage_i8(self, tmp_path: Path) -> None:
        # The real assets of karume-gemma4 (non-QAT) are schema 1 (I8, no storage field). The
        # new index always spells the storage.
        repo = stage_gemma_repo(tmp_path / "gem1", schema=1)
        out = tmp_path / "out"
        migrated(repo, out)
        read = opened(out, container_of(out, "e2b", "model", "f32"))
        index = json.loads(asset_payload(read, "ple_index"))

        assert index["schema"] == 3
        assert index["storage"] == "i8"
        assert index["values"]["rowBytes"] == PLE_LAYERS * PLE_DIM
        joined = b"".join(
            asset_payload(read, block["asset"]) for block in index["values"]["blocks"]
        )
        assert joined == source_rows(repo, "e2b/ple", "values")

    def test_the_ple_assets_leave_the_manifest_and_the_files_are_not_copied(
        self, gemma_repo: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        migrated(gemma_repo, out)
        manifest = manifest_of(out)

        assert set(manifest["models"]["e2b"]["assets"]) == {"tokenizer"}
        assert not (out / "e2b" / "ple").exists()

    def test_the_fold_reaches_every_dtype_of_the_owning_component(
        self, gemma_repo: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        migrated(gemma_repo, out)

        for dtype in ("f32", "f16"):
            read = opened(out, container_of(out, "e2b", "model", dtype))
            assert "ple_index" in read.model.assets

    def test_a_ple_index_on_a_pipeline_without_an_owner_fails_loudly(
        self, gemma_repo: Path, tmp_path: Path
    ) -> None:
        manifest = json.loads((gemma_repo / "karume.json").read_text(encoding="utf-8"))
        manifest["models"]["e2b"]["pipeline"] = "synthetic/1"
        (gemma_repo / "karume.json").write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(MigrateError, match="is not a PLE owner"):
            migrated(gemma_repo, tmp_path / "out")

    def test_a_shard_whose_metadata_disagrees_with_the_index_fails_loudly(
        self, gemma_repo: Path, tmp_path: Path
    ) -> None:
        """A mismatch between the index and `karume_ple` breeds silently wrong values (a row would
        have 2 owners)."""
        manifest = json.loads((gemma_repo / "karume.json").read_text(encoding="utf-8"))
        index_path = gemma_repo / manifest["models"]["e2b"]["assets"]["ple_index"]["path"]
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["shards"][1]["stop"] = 44
        index["shards"][2]["start"] = 44
        redeclare_asset(gemma_repo, "e2b", "ple_index", json.dumps(index).encode("utf-8"))

        with pytest.raises(MigrateError, match=r"karume_ple\.stop"):
            migrated(gemma_repo, tmp_path / "out")

    def test_a_dim_that_the_packing_cannot_divide_fails_loudly(
        self, gemma_repo: Path, tmp_path: Path
    ) -> None:
        """i4 packs 2 elements per byte (a dim that does not divide leaves the row byte count
        undefined)."""
        manifest = json.loads((gemma_repo / "karume.json").read_text(encoding="utf-8"))
        index_path = gemma_repo / manifest["models"]["e2b"]["assets"]["ple_index"]["path"]
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["dim"] = PLE_DIM - 1
        redeclare_asset(gemma_repo, "e2b", "ple_index", json.dumps(index).encode("utf-8"))

        with pytest.raises(
            MigrateError, match=r"dim 15 is not divisible by the storage .i4. pack factor 2"
        ):
            migrated(gemma_repo, tmp_path / "out")

    def test_a_row_that_is_not_a_multiple_of_four_bytes_fails_loudly(self, tmp_path: Path) -> None:
        """Blocks are cut at multiples of a row (a row that is not a multiple of 4 is incompatible
        with the alignment of the block length)."""
        root = stage_gemma_repo(tmp_path / "thin", layers=1, dim=4)  # one values row = 2 bytes

        with pytest.raises(MigrateError, match=r"one values row of 2 bytes is not a multiple of 4"):
            migrated(root, tmp_path / "out")

    @pytest.mark.parametrize(
        ("edit", "message"),
        [
            (
                lambda header: header["values"].update(dtype="X9"),
                r"Tensor 'values': dtype 'X9' unknown to the reader",
            ),
            (
                # The row count and the byte length still match; only the shape of the rows lies
                # (this passed before, when only the row count and the byte length were checked).
                lambda header: header["values"].update(
                    shape=[header["values"]["shape"][0], PLE_LAYERS, PLE_DIM * 2]
                ),
                r"Tensor 'values': size mismatch",
            ),
        ],
        ids=["unknown-dtype", "shape-disagrees-with-length"],
    )
    def test_a_shard_that_breaks_the_reader_layout_fails_loudly(
        self, gemma_repo: Path, tmp_path: Path, edit: Any, message: str
    ) -> None:
        """Goes through the same layout check as reading an old shard (`legacy.read_component`)."""
        doctor_ple_shard(gemma_repo, 2, edit)

        with pytest.raises(ContainerError, match=message):
            migrated(gemma_repo, tmp_path / "out")

    def test_a_metadata_value_that_is_not_a_string_fails_loudly(
        self, gemma_repo: Path, tmp_path: Path
    ) -> None:
        """`__metadata__` is taken verbatim (not silently passed through a `str()` conversion)."""
        doctor_ple_shard(gemma_repo, 2, lambda header: header["__metadata__"].update(note=5))

        with pytest.raises(LegacyFormatError, match="__metadata__ is not a string → string map"):
            migrated(gemma_repo, tmp_path / "out")

    def test_a_doctored_but_well_formed_shard_still_migrates(
        self, gemma_repo: Path, tmp_path: Path
    ) -> None:
        """Control — only rewriting the header (with the same declaration) passes (the checks above
        do not fail tautologically)."""
        doctor_ple_shard(gemma_repo, 2, lambda header: None)

        assert migrated(gemma_repo, tmp_path / "out").converted

    def test_an_owner_without_a_local_seat_fails_loudly(
        self, gemma_repo: Path, tmp_path: Path
    ) -> None:
        """Removing it from assets with no fold target at all would silently drop the PLE from the
        distribution."""
        lender_out = tmp_path / "lender"
        migrated(gemma_repo, lender_out)
        lender = json.loads((gemma_repo / "karume.json").read_text(encoding="utf-8"))
        root = tmp_path / "borrower"
        index, ple = stage_ple(root, "e2b/ple")
        weights = {
            dtype: {
                "shards": [
                    {**ref, "repo": "hdae/lender", "revision": "a" * 40}
                    for ref in lender["models"]["e2b"]["weights"]["model"][dtype]["shards"]
                ]
            }
            for dtype in ("f32", "f16")
        }
        write_manifest(
            root,
            {
                "e2b": model_entry(
                    {"model": weights},
                    {"ple_index": index, **ple},
                    {"full": quant({"model": "f32"})},
                    pipeline="gemma4",
                )
            },
            "e2b",
        )

        with pytest.raises(MigrateError, match="has no slot converted in this repository"):
            migrated(
                root,
                tmp_path / "out",
                cross_repos=[parse_cross_repo(f"hdae/lender={lender_out}@{REVISION}")],
            )


class TestTheCli:
    def test_it_migrates_the_repository_and_prints_a_summary(
        self, repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "out"
        migrate.main(
            ["--manifest", str(repo / "karume.json"), "--out", str(out), "--license", "apache-2.0"]
        )
        printed = capsys.readouterr().out

        assert (out / "karume.json").is_file()
        assert "containers=2" in printed and "crossed=0" in printed

    def test_a_positional_path_cannot_be_combined_with_the_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(
            MigrateError, match="--manifest cannot be combined with positional arguments"
        ):
            migrate.main(
                [
                    "a/model.safetensors",
                    "--manifest",
                    "a/karume.json",
                    "--out",
                    str(tmp_path),
                    "--license",
                    "mit",
                ]
            )

    @pytest.mark.parametrize("extra", [["--graph-name", "x"], ["--graph"]])
    def test_the_component_only_options_are_refused(self, tmp_path: Path, extra: list[str]) -> None:
        with pytest.raises(MigrateError, match="is an option of per-component mode"):
            migrate.main(
                [
                    "--manifest",
                    "a/karume.json",
                    "--out",
                    str(tmp_path),
                    "--license",
                    "mit",
                    *extra,
                ]
            )

    def test_cross_repo_is_refused_in_component_mode(self, tmp_path: Path) -> None:
        with pytest.raises(
            MigrateError, match="--cross-repo can only be used in whole-repository mode"
        ):
            migrate.main(
                [
                    "a/model.safetensors",
                    "--out",
                    str(tmp_path),
                    "--license",
                    "mit",
                    "--cross-repo",
                    f"hdae/x=/tmp/y@{REVISION}",
                ]
            )

    def test_it_requires_either_a_component_or_a_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(
            MigrateError, match="representative path of a component to migrate or --manifest"
        ):
            migrate.main(["--out", str(tmp_path), "--license", "mit"])


class TestThePartLength:
    """Whole-repository mode can choose the part length too (the writer's choice set —
    container-v1 §4.2 / ADR 0108)."""

    @pytest.fixture
    def small_parts(self) -> None:
        """Cancels the module's autouse fixture (which widens the set with the synthetic size) —
        checks with the real set."""

    @staticmethod
    def _spy(monkeypatch: pytest.MonkeyPatch) -> list[int]:
        """Collect the part lengths that reached the writer, in call order (the write goes through
        unchanged)."""
        seen: list[int] = []
        original = publish.write_model_container

        def spy(*args, **kwargs):
            seen.append(kwargs["part_bytes"])
            return original(*args, **kwargs)

        monkeypatch.setattr(publish, "write_model_container", spy)
        return seen

    @pytest.mark.parametrize(
        ("flag", "expected"), [([], 256 * MIB), (["--part-bytes", "512"], 512 * MIB)]
    )
    def test_every_container_is_written_with_the_chosen_length(
        self,
        repo: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        flag: list[str],
        expected: int,
    ) -> None:
        """The default is 256 MiB; `--part-bytes 512` reaches the writer of every container as the
        byte count of 512 MiB."""
        seen = self._spy(monkeypatch)
        out = tmp_path / "out"
        migrate.main(
            ["--manifest", str(repo / "karume.json"), "--out", str(out), "--license", "mit", *flag]
        )

        assert seen == [expected, expected]
        # The containers written with the chosen part length are declared as the
        # `container.parts` of karume/5 and can be read back.
        for component in ("encoder", "decoder"):
            container = container_of(out, "alpha", component, "f32")
            assert len(container["parts"]) >= 2
            assert opened(out, container).header.kind == "model"

    def test_the_function_default_is_256_mib(
        self, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default of the path that calls the function directly (the CLI passes the default
        explicitly, so a CLI check cannot protect it)."""
        seen = self._spy(monkeypatch)
        migrate_repository(repo / "karume.json", tmp_path / "out", provenance=PROVENANCE)

        assert seen == [256 * MIB, 256 * MIB]

    def test_a_length_outside_the_choices_fails_loudly_before_writing(
        self, repo: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"

        with pytest.raises(
            MigrateError,
            match=(
                r"Part length 314572800 bytes is outside the writer's choice set"
                r" \{256, 512, 768, 1024\} MiB"
            ),
        ):
            migrate_repository(
                repo / "karume.json", out, provenance=PROVENANCE, part_bytes=300 * MIB
            )

        assert not out.exists()


class TestThePathsInTheOldManifest:
    """The paths of the old manifest are read with the same grammar as the hub (R7-002 — nothing is
    written or read outside the output directory).

    The write target joins the old path's parent directory under `out` as is, so letting `../` or
    an absolute path through would write the new container outside the output directory (inside
    the old input's tree), and the manifest would carry a path the hub rejects. A declaration that
    points outside the input repository through a symlink is rejected too, since it reads outside
    the old input.
    """

    @staticmethod
    def _redeclare(repo: Path, path: str) -> None:
        manifest = json.loads((repo / "karume.json").read_text(encoding="utf-8"))
        for model in manifest["models"].values():
            model["assets"]["tokenizer"]["path"] = path
        (repo / "karume.json").write_text(json.dumps(manifest), encoding="utf-8")

    @pytest.mark.parametrize(
        ("spelling", "message"),
        [
            ("../outside/tokenizer.json", "leading dot"),
            ("/outside/tokenizer.json", "empty segment"),
        ],
        ids=["parent", "absolute"],
    )
    def test_a_path_outside_the_repository_is_refused_before_anything_is_written(
        self, repo: Path, tmp_path: Path, spelling: str, message: str
    ) -> None:
        # The target file does exist (checks that it is refused although it exists — as distinct
        # from failing because it is missing).
        place(tmp_path, "outside/tokenizer.json", (repo / "tokenizer.json").read_bytes())
        self._redeclare(repo, spelling)
        out = tmp_path / "out"

        with pytest.raises(MigrateError, match=message):
            migrated(repo, out)
        assert not out.exists()

    def test_a_symlink_that_leaves_the_repository_is_refused(
        self, repo: Path, tmp_path: Path
    ) -> None:
        elsewhere = tmp_path / "elsewhere"
        place(elsewhere, "tokenizer.json", (repo / "tokenizer.json").read_bytes())
        (repo / "linked").symlink_to(elsewhere, target_is_directory=True)
        self._redeclare(repo, "linked/tokenizer.json")
        out = tmp_path / "out"

        with pytest.raises(MigrateError, match="outside the input repository"):
            migrated(repo, out)
        assert not out.exists()


class TestAFailedRepositoryMigrationLeavesNothing:
    """A whole-repository migration is assembled in a staging directory and placed in the output
    directory only when everything passed (R7-011).

    Placing the components into the final output directory one by one would leave only the first
    container behind when the second component fails, and a rerun with the same arguments would
    stop with "artifacts of a previous run remain" (no progress until the output directory is
    removed by hand).
    """

    def test_a_failure_in_the_second_unit_leaves_the_output_empty_and_rerunnable(
        self, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = migrate._migrate_shards
        calls = 0

        def second_fails(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise MigrateError("a run that failed converting the second component")
            return real(*args, **kwargs)

        monkeypatch.setattr(migrate, "_migrate_shards", second_fails)
        out = tmp_path / "out"

        with pytest.raises(MigrateError, match="the second component"):
            migrated(repo, out)
        assert calls == 2, "premise of the subject: there are 2 or more units to convert"
        assert not out.exists() or not any(out.iterdir())

        monkeypatch.setattr(migrate, "_migrate_shards", real)
        migrated(repo, out)
        assert (out / "karume.json").is_file()

    def test_an_output_that_is_not_empty_is_refused_before_any_conversion(
        self, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "stray.txt").write_text("前回の残り\n", encoding="utf-8")

        def never(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a conversion ran before the output directory check")

        monkeypatch.setattr(migrate, "_migrate_shards", never)

        with pytest.raises(MigrateError, match="Artifacts of a previous run"):
            migrated(repo, out)
        assert sorted(path.name for path in out.iterdir()) == ["stray.txt"]

    def test_an_empty_output_directory_is_accepted(self, repo: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()

        migrated(repo, out)

        assert (out / "karume.json").is_file()


class TestThePleShardDeclarations:
    """The dtype and **the whole shape** of the old PLE shards are matched against the index (the
    PLE side of R7-003).

    Looking only at the leading dimension and the byte length would silently fold another shape
    with the same element count (the last 2 axes of `[rows, layers, dim]` swapped) or another type
    with the same byte length (an F32 scale declared as I32). The old writer wrote `values` with
    the old storage spelling (i8 → I8 / i4 → I4 / i2 → I2) + `[rows, layers, dim]`, and `scales`
    as F32 + `[rows, layers]`.
    """

    @pytest.mark.parametrize(
        ("key", "spec", "message"),
        [
            ("values", {"shape": [25, PLE_LAYERS * 2, PLE_DIM // 2]}, r"values.*shape"),
            ("scales", {"dtype": "I32"}, r"scales.*dtype I32"),
        ],
        ids=["values-shape", "scales-dtype"],
    )
    def test_a_declaration_that_disagrees_with_the_index_fails_loudly(
        self, gemma_repo: Path, tmp_path: Path, key: str, spec: dict[str, Any], message: str
    ) -> None:
        def edit(header: dict[str, Any]) -> None:
            assert header[key]["shape"][0] == 25, "premise of the subject: shard 2 has 25 rows"
            header[key].update(spec)

        doctor_ple_shard(gemma_repo, 2, edit)
        out = tmp_path / "out"

        with pytest.raises(MigrateError, match=message):
            migrated(gemma_repo, out)
        assert not out.exists() or not any(out.iterdir())


class TestATrailingNewlineIsRefused:
    """Spellings in the old manifest and in `--cross-repo` are refused with a trailing newline (the
    same accepted set as the hub).

    Python's `$` also matches just before a trailing newline, so looking `^…$` up with `.match`
    lets a trailing newline through.
    """

    @pytest.mark.parametrize(
        "spec",
        [f"hdae/lender\n=/tmp/x@{REVISION}", f"hdae/lender=/tmp/x@{REVISION}\n"],
    )
    def test_a_cross_repo_option_with_a_trailing_newline_fails_loudly(self, spec: str) -> None:
        with pytest.raises(MigrateError, match="--cross-repo"):
            parse_cross_repo(spec)

    def test_a_pipeline_with_a_trailing_newline_fails_loudly(
        self, repo: Path, tmp_path: Path
    ) -> None:
        manifest = json.loads((repo / "karume.json").read_text(encoding="utf-8"))
        manifest["models"]["alpha"]["pipeline"] = "synthetic/1\n"
        (repo / "karume.json").write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(MigrateError, match="does not have the form '<name>/<major>'"):
            migrated(repo, tmp_path / "out")

    def test_a_sha256_with_a_trailing_newline_fails_loudly(
        self, repo: Path, tmp_path: Path
    ) -> None:
        manifest = json.loads((repo / "karume.json").read_text(encoding="utf-8"))
        manifest["models"]["alpha"]["assets"]["tokenizer"]["sha256"] = "a" * 64 + "\n"
        (repo / "karume.json").write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(MigrateError, match=r"assets\['tokenizer'\]\.sha256 is not a sha256"):
            migrated(repo, tmp_path / "out")
